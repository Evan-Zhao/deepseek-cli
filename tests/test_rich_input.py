"""Tests for the rich interactive input handler — @ file mentions & completions."""

from __future__ import annotations

import os
from pathlib import Path

from prompt_toolkit.document import Document

from deepseek.utils.rich_input import (
    FileMentionCompleter,
    RichInputHandler,
    _find_pattern_at_mention,
    _get_path_meta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _display_text(completion) -> str:
    """Extract plain display text from a Completion (display is a
    ``FormattedText``, i.e. a list of ``(style, text)`` tuples)."""
    return "".join(segment[1] for segment in completion.display)

# ===========================================================================
# _find_pattern_at_mention — detecting @-mentions in the buffer
# ===========================================================================


class TestFindAtMention:
    def test_detects_at_prefix(self):
        doc = Document(text="@", cursor_position=1)
        assert _find_pattern_at_mention(doc) == ""

    def test_detects_at_mid_sentence(self):
        doc = Document(text="explain @src/main.py", cursor_position=20)
        assert _find_pattern_at_mention(doc) == "src/main.py"

    def test_detects_partial_path(self):
        doc = Document(text="check @src/", cursor_position=11)
        assert _find_pattern_at_mention(doc) == "src/"

    def test_returns_none_when_no_at(self):
        doc = Document(text="just a normal message", cursor_position=21)
        assert _find_pattern_at_mention(doc) is None

    def test_returns_none_after_mention_completed(self):
        """Once the user types a space after an @-mention, the completer stops."""
        doc = Document(text="see @file.txt and", cursor_position=17)
        assert _find_pattern_at_mention(doc) is None

    def test_only_latest_at_mention_detected(self):
        doc = Document(text="@old @src/main.py", cursor_position=17)
        assert _find_pattern_at_mention(doc) == "src/main.py"


# ===========================================================================
# FileMentionCompleter — completion popup logic
# ===========================================================================


class TestFileMentionCompleter:
    def test_empty_when_no_mention(self, tmp_path):
        """No @ mention in buffer → no completions."""
        completer = FileMentionCompleter()
        doc = Document(text="plain text", cursor_position=10)
        completions = list(completer.get_completions(doc, None))
        assert completions == []

    def test_lists_top_level_directory(self, tmp_path):
        """Typing @ shows entries from the CWD."""
        completer = FileMentionCompleter()
        doc = Document(text="@", cursor_position=1)
        completions = list(completer.get_completions(doc, None))
        # There should be items from the test runner's CWD
        assert len(completions) > 0

    def test_directory_contents_with_trailing_slash(self, tmp_path):
        """@dir/ lists the contents of that directory."""
        # Create files directly in tmp_path (not in a subdirectory)
        (tmp_path / "alpha.txt").write_text("a")
        (tmp_path / "beta.py").write_text("b")

        completer = FileMentionCompleter()
        text = f"@{tmp_path.name}/"
        doc = Document(text=text, cursor_position=len(text))
        # Change into the parent of tmp_path for the completer to work
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path.parent)
            completions = list(completer.get_completions(doc, None))
        finally:
            os.chdir(original_cwd)

        # Completion inserts basenames (preserving @); full path shown in display
        texts = [c.text for c in completions]
        displays = [_display_text(c) for c in completions]
        assert "alpha.txt" in texts
        assert "beta.py" in texts
        assert any(tmp_path.name + "/alpha.txt" in d for d in displays)
        assert any(tmp_path.name + "/beta.py" in d for d in displays)

    def test_nested_path_completions(self, tmp_path):
        """@a/b/ should list files inside a/b."""
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "file.txt").write_text("hello")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            completer = FileMentionCompleter()
            doc = Document(text="@a/b/", cursor_position=5)
            completions = list(completer.get_completions(doc, None))
            texts = [c.text for c in completions]
            displays = [_display_text(c) for c in completions]
            assert "file.txt" in texts
            assert any("a/b/file.txt" in d for d in displays)
        finally:
            os.chdir(original_cwd)

    def test_partial_name_filters_completions(self, tmp_path):
        """Typing @dir/fil should show only files starting with 'fil'."""
        d = tmp_path / "data"
        d.mkdir()
        (d / "file_a.txt").write_text("a")
        (d / "file_b.txt").write_text("b")
        (d / "notes.txt").write_text("c")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            completer = FileMentionCompleter()
            doc = Document(text="@data/file", cursor_position=10)
            completions = list(completer.get_completions(doc, None))
            texts = [c.text for c in completions]
            assert any("file_a" in t for t in texts)
            assert any("file_b" in t for t in texts)
            assert not any("notes" in t for t in texts)
        finally:
            os.chdir(original_cwd)

    def test_completions_insert_basename_keep_at(self, tmp_path):
        """Completion inserts only the basename (not the full path) so that
        the @ symbol and any directory prefix in the buffer are preserved."""
        d = tmp_path / "src"
        d.mkdir()
        (d / "main.py").write_text("x")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            completer = FileMentionCompleter()
            doc = Document(text=f"@src/main.py", cursor_position=11)
            completions = list(completer.get_completions(doc, None))
            if completions:
                c = completions[0]
                assert c.text == "main.py", f"Expected basename, got {c.text!r}"
                assert "src/main.py" in _display_text(c), (
                    f"Expected full path in display, got {_display_text(c)!r}"
                )
        finally:
            os.chdir(original_cwd)

    def test_directories_show_trailing_slash(self, tmp_path):
        """Directory completions should include a trailing / in both text and display."""
        d = tmp_path / "mydir"
        d.mkdir()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            completer = FileMentionCompleter()
            doc = Document(text="@", cursor_position=1)
            completions = list(completer.get_completions(doc, None))
            dir_completions = [c for c in completions if _display_text(c).rstrip().endswith("/")]
            assert any("mydir/" in _display_text(c) for c in dir_completions)
        finally:
            os.chdir(original_cwd)

    def test_max_completions_respected(self, tmp_path):
        """Should not exceed max_completions limit."""
        d = tmp_path / "many"
        d.mkdir()
        for i in range(100):
            (d / f"file_{i:03d}.txt").write_text("x")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            completer = FileMentionCompleter(max_completions=10)
            doc = Document(text="@many/", cursor_position=6)
            completions = list(completer.get_completions(doc, None))
            assert len(completions) <= 10
        finally:
            os.chdir(original_cwd)

    def test_nonexistent_directory_returns_empty(self):
        """Completing inside a non-existent directory should yield no results."""
        completer = FileMentionCompleter()
        doc = Document(text="@/nonexistent_dir_xyz/", cursor_position=22)
        completions = list(completer.get_completions(doc, None))
        assert completions == []

    def test_permission_error_returns_empty(self, tmp_path):
        """An unreadable directory should not crash the completer."""
        restricted = tmp_path / "secret"
        restricted.mkdir(mode=0o000)
        try:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                completer = FileMentionCompleter()
                doc = Document(text="@secret/", cursor_position=8)
                completions = list(completer.get_completions(doc, None))
                # Should be empty (PermissionError caught), not crash
                assert completions == []
            finally:
                os.chdir(original_cwd)
        finally:
            # Restore permissions so cleanup can delete the directory
            restricted.chmod(0o755)


# ===========================================================================
# _get_path_meta — file metadata text for completion display
# ===========================================================================


class TestGetPathMeta:
    def test_file_shows_size_in_bytes(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("hello")
        meta = _get_path_meta(str(f))
        assert "B" in meta

    def test_directory_shows_directory_label(self, tmp_path):
        meta = _get_path_meta(str(tmp_path))
        assert "directory" in meta

    def test_nonexistent_path_returns_question_mark(self):
        meta = _get_path_meta("/dev/null/nonexistent_path_xyz")
        assert meta == "?"

    def test_larger_file_shows_kb(self, tmp_path):
        f = tmp_path / "medium.txt"
        f.write_text("x" * 2048)
        meta = _get_path_meta(str(f))
        assert "KB" in meta

    def test_large_file_shows_mb(self, tmp_path):
        f = tmp_path / "large.txt"
        f.write_text("x" * 2_000_000)
        meta = _get_path_meta(str(f))
        assert "MB" in meta


# ===========================================================================
# RichInputHandler — _process_mentions (non-interactive part)
# ===========================================================================


class TestRichInputHandlerProcessMentions:
    def test_no_mention_returns_text_unchanged(self):
        handler = RichInputHandler()
        result = handler._process_mentions("hello world")
        assert result == "hello world"

    def test_nonexistent_file_mention_left_as_is(self):
        handler = RichInputHandler()
        result = handler._process_mentions("check @nonexistent_file_xyz.txt")
        assert "nonexistent_file_xyz.txt" in result

    def test_real_file_attached_and_expanded(self, tmp_path):
        from deepseek.handlers.file_handler import FileHandler

        f = tmp_path / "code.py"
        f.write_text("print(42)", encoding="utf-8")

        fh = FileHandler()
        handler = RichInputHandler(file_handler=fh)
        result = handler._process_mentions(f"explain @{f}")

        # The file should be attached
        assert len(fh.list_attachments()) == 1
        # The @-reference should be replaced
        assert "📎" in result
        assert fh.attached_files[0]["content"] == "print(42)"
