import json
import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path
import html

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Common question/answer field name pairs to try in auto mode, in priority order.
_AUTO_PAIRS = [
    ("Front", "Back"),
    ("Question", "Answer"),
    ("Prompt", "Answer"),
    ("Term", "Definition"),
    ("Definition", "Word"),
]


def _strip_html(text: str) -> str:
    text = html.unescape(text)
    return _HTML_TAG_RE.sub("", text).strip()


def _load_field_names(conn: sqlite3.Connection) -> dict[int, list[str]]:
    """Return {mid: [field_name, ...]} from the col.models JSON blob."""
    row = conn.execute("SELECT models FROM col").fetchone()
    if not row:
        return {}
    models = json.loads(row[0])
    result = {}
    for mid_str, model in models.items():
        names = [f["name"] for f in sorted(model["flds"], key=lambda f: f["ord"])]
        result[int(mid_str)] = names
    return {}.__class__(result)  # plain dict


def _fields_as_dict(flds_str: str, field_names: list[str]) -> dict[str, str]:
    """Zip raw field values with their names, strip HTML from each."""
    raw_values = flds_str.split("\x1f")
    return {
        name: _strip_html(raw_values[i]) if i < len(raw_values) else ""
        for i, name in enumerate(field_names)
    }


def _pick_question_answer(
    fields: dict[str, str],
    question_field: str | None,
    answer_field: str | None,
) -> tuple[str, str] | None:
    """
    Return (question, answer) or None if the card should be skipped.

    Manual mode: use the caller-supplied field names.
    Auto mode: try _AUTO_PAIRS, then fall back to first two non-empty fields.
    """
    available = list(fields.keys())

    if question_field is not None and answer_field is not None:
        # Manual mode — caller already validated names exist
        q = fields.get(question_field, "")
        a = fields.get(answer_field, "")
        return (q, a) if q and a else None

    # Auto mode — try known pairs
    for q_name, a_name in _AUTO_PAIRS:
        if q_name in fields and a_name in fields:
            q, a = fields[q_name], fields[a_name]
            if q and a:
                return q, a

    # Auto fallback — first two non-empty fields
    non_empty = [v for v in (fields.get(n, "") for n in available) if v]
    if len(non_empty) >= 2:
        return non_empty[0], non_empty[1]

    return None


def inspect_deck_fields(apkg_path: str) -> dict[str, list[str]]:
    """
    Return {model_name: [field_name, ...]} for every note type in the deck.
    Useful for choosing --question-field / --answer-field values.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        raise FileNotFoundError(f"Deck not found: {apkg_path}")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(apkg_path, "r") as zf:
            zf.extractall(tmp)

        db_path = _find_db(tmp)
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT models FROM col").fetchone()
            if not row:
                return {}
            models = json.loads(row[0])

    return {
        model["name"]: [f["name"] for f in sorted(model["flds"], key=lambda f: f["ord"])]
        for model in models.values()
    }


def _find_db(tmp_dir: str) -> Path:
    for name in ("collection.anki21", "collection.anki2"):
        p = Path(tmp_dir) / name
        if p.exists():
            return p
    raise ValueError("No collection.anki2 or collection.anki21 found inside the .apkg file")


def load_deck(
    apkg_path: str,
    question_field: str | None = None,
    answer_field: str | None = None,
    include_fields: bool = False,
) -> list[dict]:
    """
    Load an Anki .apkg and return cards as [{'question': str, 'answer': str}, ...].

    question_field / answer_field — manual field name selection.
      If one is provided, both must be provided.
      Raises ValueError with available field names if a name doesn't exist.

    include_fields — if True, each card also contains a 'fields' dict with
      all field values for debugging.
    """
    if (question_field is None) != (answer_field is None):
        raise ValueError("Provide both --question-field and --answer-field, or neither.")

    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        raise FileNotFoundError(f"Deck not found: {apkg_path}")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(apkg_path, "r") as zf:
            zf.extractall(tmp)

        db_path = _find_db(tmp)

        with sqlite3.connect(str(db_path)) as conn:
            mid_to_names = _load_field_names(conn)

            # Validate manual field names against all known field names in the deck
            if question_field is not None:
                all_names: set[str] = set()
                for names in mid_to_names.values():
                    all_names.update(names)
                for supplied in (question_field, answer_field):
                    if supplied not in all_names:
                        raise ValueError(
                            f"Field {supplied!r} not found in deck.\n"
                            f"Available fields: {sorted(all_names)}"
                        )

            cards = []
            cursor = conn.execute("SELECT mid, flds FROM notes")
            for mid, flds_str in cursor:
                field_names = mid_to_names.get(mid)
                if not field_names:
                    # Fallback: positional split (old behaviour)
                    parts = flds_str.split("\x1f")
                    if len(parts) < 2:
                        continue
                    q, a = _strip_html(parts[0]), _strip_html(parts[1])
                    if q and a:
                        card: dict = {"question": q, "answer": a}
                        if include_fields:
                            card["fields"] = {str(i): v for i, v in enumerate(parts)}
                        cards.append(card)
                    continue

                fields = _fields_as_dict(flds_str, field_names)
                pair = _pick_question_answer(fields, question_field, answer_field)
                if pair is None:
                    continue
                q, a = pair
                card = {"question": q, "answer": a}
                if include_fields:
                    card["fields"] = fields
                cards.append(card)

    return cards
