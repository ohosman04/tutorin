import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path
import html

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    text = html.unescape(text)  
    return _HTML_TAG_RE.sub("", text).strip()

def load_deck(apkg_path: str) -> list[dict]:
    """
    Unpack an Anki .apkg file and return its cards as a list of
    {'question': str, 'answer': str} dicts.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        raise FileNotFoundError(f"Deck not found: {apkg_path}")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(apkg_path, "r") as zf:
            zf.extractall(tmp)

        # Prefer modern Anki databases first
        db_candidates = [
            Path(tmp) / "collection.anki21",
            Path(tmp) / "collection.anki2",
        ]

        db_path = next((p for p in db_candidates if p.exists()), None)

        if db_path is None:
            raise ValueError(
                "No collection.anki2 or collection.anki21 found inside the .apkg file"
            )

        cards = []
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT flds FROM notes")
            for (flds,) in cursor:
                parts = flds.split("\x1f")
                if len(parts) < 2:
                    continue
                question = _strip_html(parts[0])
                answer = _strip_html(parts[1])
                if question and answer:
                    cards.append({"question": question, "answer": answer})

    return cards
