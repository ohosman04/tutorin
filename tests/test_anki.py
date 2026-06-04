import sqlite3
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.anki_parser import load_deck


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_apkg(cards: list[tuple[str, str]]) -> Path:
    """
    Build a minimal .apkg file in a temp directory and return its path.
    The caller is responsible for cleanup (use tmp_path fixture).
    """
    tmp = Path(tempfile.mkdtemp())
    db_path = tmp / "collection.anki2"

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE notes (flds TEXT)")
        for front, back in cards:
            conn.execute("INSERT INTO notes VALUES (?)", (f"{front}\x1f{back}",))
        conn.commit()

    apkg_path = tmp / "deck.apkg"
    with zipfile.ZipFile(apkg_path, "w") as zf:
        zf.write(db_path, "collection.anki2")

    return apkg_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDeck:
    def test_basic_cards_returned(self, tmp_path):
        apkg = _make_apkg([
            ("What is Python?", "A high-level programming language."),
            ("What is a list?", "An ordered, mutable sequence."),
        ])
        cards = load_deck(str(apkg))

        assert len(cards) == 2
        assert cards[0] == {
            "question": "What is Python?",
            "answer": "A high-level programming language.",
        }
        assert cards[1] == {
            "question": "What is a list?",
            "answer": "An ordered, mutable sequence.",
        }

    def test_html_tags_stripped(self, tmp_path):
        apkg = _make_apkg([
            ("<b>Capital of France?</b>", "<div><b>Paris</b></div>"),
        ])
        cards = load_deck(str(apkg))

        assert cards[0]["question"] == "Capital of France?"
        assert cards[0]["answer"] == "Paris"

    def test_notes_with_fewer_than_two_fields_skipped(self, tmp_path):
        """A flds value with no \x1f separator should be silently skipped."""
        apkg = _make_apkg([])

        # Manually inject a malformed row
        db_path = Path(tempfile.mkdtemp()) / "collection.anki2"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE notes (flds TEXT)")
            conn.execute("INSERT INTO notes VALUES (?)", ("only_front",))
            conn.execute(
                "INSERT INTO notes VALUES (?)",
                ("Good question\x1fGood answer",),
            )
            conn.commit()

        apkg_path = Path(tempfile.mkdtemp()) / "deck.apkg"
        with zipfile.ZipFile(apkg_path, "w") as zf:
            zf.write(db_path, "collection.anki2")

        cards = load_deck(str(apkg_path))
        assert len(cards) == 1
        assert cards[0]["question"] == "Good question"

    def test_empty_deck_returns_empty_list(self, tmp_path):
        apkg = _make_apkg([])
        assert load_deck(str(apkg)) == []

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_deck(str(tmp_path / "nonexistent.apkg"))

    def test_missing_db_inside_apkg_raises(self, tmp_path):
        """An .apkg with no collection.anki2 should raise ValueError."""
        apkg_path = tmp_path / "empty.apkg"
        with zipfile.ZipFile(apkg_path, "w") as zf:
            zf.writestr("README.txt", "nothing here")

        with pytest.raises(ValueError, match="collection.anki2"):
            load_deck(str(apkg_path))


class TestLoadDeckMocked:
    """
    Unit tests that mock the sqlite3 connection so no real file I/O happens.
    Useful for fast CI and verifying the parsing / HTML-stripping logic in isolation.
    """

    def _run_with_rows(self, rows: list[tuple[str]]):
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter(rows))

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_cursor

        mock_zip_file = MagicMock()
        mock_zip_file.__enter__ = MagicMock(return_value=mock_zip_file)
        mock_zip_file.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.anki_parser.zipfile.ZipFile", return_value=mock_zip_file),
            patch("src.anki_parser.sqlite3.connect", return_value=mock_conn),
            patch("src.anki_parser.Path.exists", return_value=True),
            patch("src.anki_parser.tempfile.TemporaryDirectory") as mock_tmp,
        ):
            mock_tmp.return_value.__enter__ = MagicMock(return_value="/fake/tmp")
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

            return load_deck("fake.apkg")

    def test_mocked_plain_text_card(self):
        cards = self._run_with_rows([
            ("Hello\x1fWorld",),
        ])
        assert cards == [{"question": "Hello", "answer": "World"}]

    def test_mocked_html_stripped(self):
        cards = self._run_with_rows([
            ("<div><b>Front</b></div>\x1f<i>Back</i>",),
        ])
        assert cards[0] == {"question": "Front", "answer": "Back"}

    def test_mocked_multiple_cards(self):
        rows = [(f"Q{i}\x1fA{i}",) for i in range(5)]
        cards = self._run_with_rows(rows)
        assert len(cards) == 5
        assert cards[2] == {"question": "Q2", "answer": "A2"}
