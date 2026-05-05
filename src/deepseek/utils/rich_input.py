"""Rich interactive input handler for DeepSeek CLI.

Provides a prompt_toolkit-based input experience with:
  - ``@``-triggered file mention completions (like VS Code / Claude Code)
  - Auto-attach of mentioned files via FileHandler
  - Multi-line editing with configurable submit behaviour
  - Syntax-highlighted prompts via ``rich``
"""

import os
import re
from pathlib import Path
from typing import Callable, List, Optional

import prompt_toolkit.input.ansi_escape_sequences as _ptk_ansi
import prompt_toolkit.key_binding.key_bindings as _ptk_kb
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
)
from prompt_toolkit.document import Document
from prompt_toolkit.filters import has_completions
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.styles import Style
from rich.console import Console

from deepseek.handlers.file_handler import FileHandler

# ---------------------------------------------------------------------------
# Monkey-patch prompt_toolkit to support Shift+Enter as a distinct key
# ---------------------------------------------------------------------------
# prompt_toolkit's Keys enum has no ShiftEnter member, so ``_parse_key``
# rejects "s-enter" and the vt100 parser remaps the CSI u escape sequence
# ``\x1b[27;2;13~`` to plain ``Keys.ControlM``.
#
# The upstream PR https://github.com/prompt-toolkit/python-prompt-toolkit/pull/2040
# adds ``Keys.ShiftEnter`` and corrects the ANSI mapping.  Until that PR is
# merged and released we apply the same two changes at runtime.

# 1. Teach _parse_key to accept known extended key names.
_orig_parse_key = _ptk_kb._parse_key
_EXTENDED_KEYS = {"s-enter", "c-enter", "c-s-enter"}


def _patched_parse_key(key):
    if key in _EXTENDED_KEYS:
        return key
    return _orig_parse_key(key)


_ptk_kb._parse_key = _patched_parse_key  # type: ignore

# 2. Remap CSI u sequences for modified Enter from plain Enter to the
#    correct key names so the vt100 parser creates distinguishable
#    KeyPress events.
_ptk_ansi.ANSI_SEQUENCES.update(  # type: ignore
    {
        "\x1b[27;2;13~": "s-enter",  # Shift + Enter
        "\x1b[27;5;13~": "c-enter",  # Ctrl + Enter
        "\x1b[27;6;13~": "c-s-enter",  # Ctrl + Shift + Enter
    }
)


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


def _find_pattern_at_mention(document: Document) -> Optional[str]:
    """If the cursor is inside or right after an ``@mention``, return
    ``pattern`` that can be fed to a Completer.
    Returns ``None`` if we're not in a mention context.
    """
    text_before_cursor = document.text_before_cursor
    match = AT_MENTION_RE.search(text_before_cursor)
    # match.group(1) is '@', match.group(2) is the partial path
    return match.group(2) if match else None


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

        pattern = _find_pattern_at_mention(document)
        if pattern is None:
            return result

        # Determine the base directory and the partial name.
        # Support paths like @src/main -> base=src/, partial=main
        # NOTE: pathlib normalises trailing slashes away, and that is not what we want:
        #   Path("src/") == Path("src")
        # so we do this:
        if pattern.endswith("/") or pattern.endswith("\\"):
            search_path = Path(pattern)
            partial = ""
        else:
            pat_path = Path(pattern)
            search_path = pat_path.parent
            partial = pat_path.name

        # Resolve the base directory — expand ~ if configured.
        if self.expanduser:
            search_path = search_path.expanduser()

        # Gather matching entries.
        entries = sorted(search_path.glob(partial + "*"), key=lambda e: e.name)
        # Filter and build completions.
        matched = 0
        for entry in entries:
            if matched >= self.max_completions:
                break
            # Reconstruction of the full relative path for the completion
            # display in the dropdown. The inserted text is only the entry's
            # basename so the @ symbol and any directory prefix already in
            # the buffer are preserved.
            rel_path = str(entry)
            if entry.is_dir() and self.include_dirs:
                result.append(
                    Completion(
                        entry.name + "/",
                        start_position=-len(partial),
                        display=f"{rel_path}/ ",
                        display_meta="📁 directory",
                        style="fg:#4a9eff",
                    )
                )
                matched += 1
            elif entry.is_file():
                display_meta = _get_path_meta(str(entry))
                result.append(
                    Completion(
                        entry.name,
                        start_position=-len(partial),
                        display=rel_path,
                        display_meta=display_meta,
                    )
                )
                matched += 1
        return result


def _get_path_meta(path_str: str) -> str:
    """Return a short metadata string for a file (size, type)."""
    path = Path(path_str)
    if path.is_dir():
        return "📁 directory"
    try:
        size = path.stat().st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"
    except OSError:
        size_str = "?"
    return size_str


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
        submit_mode: str = "shift-enter",
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
            result = self._session.prompt("")
            text = str(result)
        except KeyboardInterrupt:
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

        @bindings.add("enter", filter=has_completions)
        def _accept_completion(event: KeyPressEvent):
            buf = event.current_buffer
            cs = buf.complete_state
            if cs is not None and cs.current_completion is not None:
                buf.apply_completion(cs.current_completion)

        if self.submit_mode == "shift-enter":

            @bindings.add("enter", filter=~has_completions)
            def _newline(event: KeyPressEvent):
                buf = event.current_buffer
                if _is_command(buf.document.text):
                    buf.validate_and_handle()
                else:
                    buf.insert_text("\n")

            @bindings.add("s-enter")
            def _submit_shift(event: KeyPressEvent):
                event.current_buffer.validate_and_handle()

        else:  # empty-line mode (default)

            @bindings.add("enter", filter=~has_completions)
            def _enter_or_submit(event: KeyPressEvent):
                buf = event.current_buffer
                text = buf.document.text
                if _is_command(text) or buf.document.current_line.strip() == "":
                    buf.validate_and_handle()
                else:
                    buf.insert_text("\n")

        # Escape key to cancel completions
        @bindings.add("escape")
        def _cancel_completions(event: KeyPressEvent):
            event.current_buffer.cancel_completion()

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
# Helper
# ---------------------------------------------------------------------------


def _is_command(text: str) -> bool:
    """Return ``True`` if *text* looks like a CLI command.

    Commands are either a leading ``/`` (e.g. ``/help``, ``/clear``) or
    the bare words ``quit``/``exit``.  In all such cases pressing ENTER
    should submit immediately rather than inserting a newline.
    """
    first_line = text.lstrip().split("\n", 1)[0]
    stripped = first_line.strip()
    return stripped.startswith("/")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def rich_input(
    prompt_text: str = "> You",
    file_handler: Optional[FileHandler] = None,
    multiline: bool = False,
    submit_mode: str = "shift-enter",
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
