"""Rich interactive input handler for DeepSeek CLI.

Provides a prompt_toolkit-based input experience with:
  - ``@``-triggered file mention completions (like VS Code / Claude Code)
  - Auto-attach of mentioned files via FileHandler
  - Multi-line editing with configurable submit behaviour
  - Syntax-highlighted prompts via ``rich``
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
)
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console

from deepseek.handlers.file_handler import FileHandler

console = Console()

# ---------------------------------------------------------------------------
# Style for the input prompt
# ---------------------------------------------------------------------------

INPUT_STYLE = Style.from_dict(
    {
        "status-toolbar": "bg:#222222 #ffffff",
        "mention": "bg:#4a9eff #ffffff",
        "completion-menu.completion": "bg:#1a1a2e #ffffff",
        "completion-menu.completion.current": "bg:#4a9eff #ffffff",
        "completion-menu.meta": "bg:#16213e #aaaaaa",
        "completion-menu.meta.current": "bg:#4a9eff #ffffff",
        "scrollbar.arrow": "bg:#4a9eff #ffffff",
        "scrollbar": "bg:#1a1a2e",
    }
)

# ---------------------------------------------------------------------------
# Regex to detect @-mentions in the buffer
# ---------------------------------------------------------------------------
# Matches @ followed by optional path characters (word chars, /, ., -, _, ~)
AT_MENTION_RE = re.compile(r"(?:^|\s)(@)([^\s@]*)$")


def _find_at_mention(document: Document) -> Optional[Tuple[int, str]]:
    """If the cursor is inside or right after an ``@mention``, return
    ``(start_position, pattern)`` that can be fed to a Completer.

    Returns ``None`` if we're not in a mention context.
    """
    text_before_cursor = document.text_before_cursor
    match = AT_MENTION_RE.search(text_before_cursor)
    if match:
        # match.group(1) is '@', match.group(2) is the partial path
        start_pos = len(match.group(1))  # how far back from cursor the word starts
        pattern = match.group(2)
        return start_pos, pattern
    return None


# ---------------------------------------------------------------------------
# File mention completer
# ---------------------------------------------------------------------------


class FileMentionCompleter(Completer):
    """Completes file paths when the user types ``@``.

    Shows files and directories from the current working directory (and
    subdirectories), filtered by the partial path after ``@``.
    """

    def __init__(
        self,
        glob_patterns: bool = True,
        expanduser: bool = True,
        max_completions: int = 30,
        include_dirs: bool = True,
    ) -> None:
        self.glob_patterns = glob_patterns
        self.expanduser = expanduser
        self.max_completions = max_completions
        self.include_dirs = include_dirs

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> List[Completion]:
        result: List[Completion] = []

        mention = _find_at_mention(document)
        if mention is None:
            return result

        start_pos, pattern = mention

        # Determine the base directory and the partial name
        # Support paths like @src/main -> base=src/, partial=main
        if "/" in pattern or "\\" in pattern:
            base_dir = os.path.dirname(pattern)
            partial = os.path.basename(pattern)
        else:
            base_dir = ""
            partial = pattern

        # Resolve the base directory for searching
        search_dir = os.path.expanduser(base_dir) if self.expanduser else base_dir
        if search_dir and not os.path.isdir(search_dir):
            # The base directory doesn't exist yet; try to find it as a partial
            # This handles cases like @src/ma where src exists but ma is partial
            parent = os.path.dirname(search_dir)
            partial = (
                os.path.basename(search_dir) + "/" + partial
                if partial
                else os.path.basename(search_dir)
            )
            search_dir = parent if parent else "."

        # Expand user home
        if self.expanduser and "~" in partial:
            partial = os.path.expanduser(partial)

        # Gather matching paths
        try:
            entries: List[str] = []
            if search_dir and os.path.isdir(search_dir):
                try:
                    entries = os.listdir(search_dir)
                except PermissionError:
                    entries = []
            elif not search_dir or search_dir == ".":
                entries = os.listdir(".")
            elif search_dir and os.path.isdir(os.path.dirname(search_dir)):
                try:
                    entries = [os.path.basename(search_dir)]
                except PermissionError:
                    entries = []

            # Filter and build completions
            partial_lower = partial.lower()
            matched = 0

            for entry in sorted(entries):
                if matched >= self.max_completions:
                    break

                entry_lower = entry.lower()
                if partial_lower and not entry_lower.startswith(partial_lower):
                    # Fuzzy: check if partial is a substring
                    if partial_lower not in entry_lower:
                        continue

                full_path = os.path.join(search_dir, entry) if search_dir else entry
                is_dir = os.path.isdir(
                    os.path.expanduser(full_path) if self.expanduser else full_path
                )

                if is_dir and self.include_dirs:
                    display = f"{full_path}/"
                    completion_text = f"{full_path}/"
                    result.append(
                        Completion(
                            completion_text,
                            start_position=-start_pos - len(partial),
                            display=display,
                            display_meta="📁 directory",
                            style="fg:#4a9eff",
                        )
                    )
                    matched += 1
                elif not is_dir:
                    # Check if it's a likely text file (skip binaries)
                    ext = Path(entry).suffix.lower()
                    binary_exts = {
                        ".exe",
                        ".dll",
                        ".so",
                        ".dylib",
                        ".bin",
                        ".obj",
                        ".o",
                        ".a",
                        ".lib",
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".bmp",
                        ".ico",
                        ".pyc",
                        ".pyo",
                        ".class",
                        ".jar",
                        ".zip",
                        ".tar",
                        ".gz",
                        ".rar",
                        ".7z",
                    }
                    if ext in binary_exts:
                        continue

                    display = full_path
                    display_meta = _get_file_meta(full_path)
                    result.append(
                        Completion(
                            full_path,
                            start_position=-start_pos - len(partial),
                            display=display,
                            display_meta=display_meta,
                        )
                    )
                    matched += 1
        except OSError:
            pass

        return result


def _get_file_meta(path: str) -> str:
    """Return a short metadata string for a file (size, type)."""
    try:
        size = os.path.getsize(path)
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"
    except OSError:
        size_str = "?"

    # Guess file type from extension
    ext = Path(path).suffix.lower()
    lang_map = {
        ".py": "🐍 Python",
        ".js": "🟨 JS",
        ".ts": "🔵 TS",
        ".tsx": "⚛️ TSX",
        ".jsx": "⚛️ JSX",
        ".rs": "🦀 Rust",
        ".go": "🔷 Go",
        ".java": "☕ Java",
        ".c": "⚙️ C",
        ".cpp": "⚙️ C++",
        ".h": "📐 Header",
        ".hpp": "📐 Header",
        ".rb": "💎 Ruby",
        ".php": "🐘 PHP",
        ".swift": "🟠 Swift",
        ".kt": "🟣 Kotlin",
        ".scala": "🔶 Scala",
        ".sh": "📜 Shell",
        ".bash": "📜 Bash",
        ".zsh": "📜 Zsh",
        ".ps1": "📜 PowerShell",
        ".md": "📝 Markdown",
        ".rst": "📝 RST",
        ".txt": "📄 Text",
        ".json": "📋 JSON",
        ".yaml": "📋 YAML",
        ".yml": "📋 YAML",
        ".toml": "📋 TOML",
        ".ini": "📋 INI",
        ".cfg": "📋 Config",
        ".xml": "📋 XML",
        ".html": "🌐 HTML",
        ".css": "🎨 CSS",
        ".scss": "🎨 SCSS",
        ".less": "🎨 LESS",
        ".sql": "🗃️ SQL",
        ".gitignore": "🙈 Git",
        ".dockerfile": "🐳 Docker",
        "dockerfile": "🐳 Docker",
        ".makefile": "🔨 Make",
        "makefile": "🔨 Make",
    }
    label = lang_map.get(ext.lower(), "📄 File")

    return f"{label} · {size_str}"


# ---------------------------------------------------------------------------
# Main input handler
# ---------------------------------------------------------------------------


class RichInputHandler:
    """Enhanced input handler with ``@`` file mention support.

    Usage::

        handler = RichInputHandler(file_handler=my_file_handler)
        user_text = handler.prompt("> You")

    After the user submits, any ``@``-mentioned files are automatically
    attached via the provided ``FileHandler``, and the ``@`` references
    are replaced with the actual file paths in the returned text.
    """

    def __init__(
        self,
        file_handler: Optional[FileHandler] = None,
        multiline: bool = False,
        submit_mode: str = "empty-line",
        mention_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Args:
            file_handler: A FileHandler to auto-attach @-mentioned files.
                If ``None``, mentions are parsed but not attached.
            multiline: Enable multi-line input.
            submit_mode: ``"shift-enter"`` or ``"empty-line"``.
            mention_callback: Optional callback invoked with the file path
                each time a file is auto-attached via ``@`` (e.g. for
                showing a confirmation message).
        """
        self.file_handler = file_handler
        self.multiline = multiline
        self.submit_mode = submit_mode
        self.mention_callback = mention_callback

        self._session: Optional[PromptSession] = None

    def prompt(self, prompt_text: str = "> You") -> str:
        """Show the input prompt and return the submitted text.

        ``@``-mentioned files are auto-attached and the ``@`` references
        are expanded to file paths in the returned text.
        """
        bindings = self._build_key_bindings()

        self._session = PromptSession(
            multiline=self.multiline,
            key_bindings=bindings,
            completer=FileMentionCompleter(),
            complete_while_typing=True,
            style=INPUT_STYLE,
        )

        # Print the prompt label with Rich (which understands [bold magenta] markup),
        # then use a plain prompt string for prompt_toolkit.
        console.print(f"[bold magenta]{prompt_text}[/bold magenta]: ", end="")

        try:
            text = str(self._session.prompt(""))
        except (KeyboardInterrupt, EOFError):
            console.print()  # ensure we move to a fresh line
            return ""

        text = text.strip()
        if not text:
            return ""

        # Process @-mentions: attach files and expand references
        return self._process_mentions(text)

    def _build_key_bindings(self) -> KeyBindings:
        """Set up enter / submit key bindings matching the existing CLI."""
        bindings = KeyBindings()

        if self.submit_mode == "shift-enter":

            @bindings.add("enter")
            def _newline(event):
                event.current_buffer.insert_text("\n")

            @bindings.add("s-enter")
            def _submit_shift(event):
                event.current_buffer.validate_and_handle()

        else:  # empty-line mode (default)

            @bindings.add("enter")
            def _enter_or_submit(event):
                buf = event.current_buffer
                if buf.document.current_line.strip() == "":
                    buf.validate_and_handle()
                else:
                    buf.insert_text("\n")

        # Ctrl+D always submits
        @bindings.add("c-d")
        def _ctrl_d(event):
            event.current_buffer.validate_and_handle()

        # Escape key to cancel completions
        @bindings.add("escape")
        def _cancel_completions(event):
            event.current_buffer.cancel_completions()

        return bindings

    def _process_mentions(self, text: str) -> str:
        """Find ``@``-mentions in *text*, attach files, and expand refs."""
        if "@" not in text:
            return text

        # Find all @-mention patterns
        mention_pattern = re.compile(r"(?:^|\s)@(\S+)")
        expanded = text
        offset = 0  # Track position shifts from replacements

        for match in mention_pattern.finditer(text):
            raw_mention = match.group(0)  # includes leading space + @
            ref = match.group(1)  # the path after @

            # Resolve the path
            expanded_path = os.path.expanduser(ref)
            abs_path = os.path.abspath(expanded_path)

            # Try to attach the file if we have a FileHandler
            if self.file_handler and os.path.isfile(abs_path):
                attached, errors = self.file_handler.attach(abs_path)
                if attached:
                    if self.mention_callback:
                        self.mention_callback(abs_path)
                    # Replace the @reference with the path in brackets
                    replacement = f"📎 `{ref}`"
                    start = match.start() + offset
                    end = match.end() + offset
                    expanded = expanded[:start] + replacement + expanded[end:]
                    offset += len(replacement) - len(raw_mention)
                elif errors:
                    # File couldn't be attached; leave the @reference as-is but notify
                    console.print(f"[yellow]⚠ Could not attach {ref}: {errors[0]}[/yellow]")
            elif os.path.isfile(abs_path):
                # No FileHandler but file exists — just expand the path in text
                replacement = f"`{ref}`"
                start = match.start() + offset
                end = match.end() + offset
                expanded = expanded[:start] + replacement + expanded[end:]
                offset += len(replacement) - len(raw_mention)
            elif os.path.isdir(abs_path):
                # Directory mentioned — keep the reference but note it
                console.print(f"[yellow]⚠ @{ref} is a directory. Use a specific file.[/yellow]")
            # If the file doesn't exist, keep the @reference as-is

        return expanded

    def attach_from_mentions(self, text: str) -> List[str]:
        """Parse ``@`` mentions from *text* and attach files, returning
        the list of successfully attached paths. Does NOT modify the text.

        This is useful if you want to attach files without expanding
        mentions in the user message.
        """
        attached: List[str] = []
        if not self.file_handler or "@" not in text:
            return attached

        mention_pattern = re.compile(r"(?:^|\s)@(\S+)")
        for match in mention_pattern.finditer(text):
            ref = match.group(1)
            expanded_path = os.path.expanduser(ref)
            abs_path = os.path.abspath(expanded_path)
            if os.path.isfile(abs_path):
                paths, _ = self.file_handler.attach(abs_path)
                attached.extend(paths)

        return attached


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def rich_input(
    prompt_text: str = "> You",
    file_handler: Optional[FileHandler] = None,
    multiline: bool = False,
    submit_mode: str = "empty-line",
) -> str:
    """One-shot rich input with ``@`` file mentions.

    Example::

        text = rich_input(\"> You\", file_handler=my_handler, multiline=True)
    """
    handler = RichInputHandler(
        file_handler=file_handler,
        multiline=multiline,
        submit_mode=submit_mode,
        mention_callback=lambda p: console.print(f"[green]📎 Attached:[/green] {p}"),
    )
    return handler.prompt(prompt_text)
