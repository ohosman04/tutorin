import hashlib
import json
import logging
import os
import random

logger = logging.getLogger(__name__)


def card_id(card: dict) -> str:
    """Stable 16-char hex ID derived from card content."""
    key = card["question"] + "\x1f" + card["answer"]
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def session_path_for(deck_path: str) -> str:
    return os.path.abspath(deck_path) + ".session.json"


def load_session(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data.get("card_order"), list):
            raise ValueError("missing card_order")
        if not isinstance(data.get("next_index"), int):
            raise ValueError("missing next_index")
        if not isinstance(data.get("retry_queue"), list):
            raise ValueError("missing retry_queue")
        return data
    except Exception as exc:
        logger.warning("Ignoring corrupt session file %s: %s", path, exc)
        return None


def save_session(path: str, state: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError as exc:
        logger.warning("Could not save session to %s: %s", path, exc)


def new_session(cards: list[dict]) -> dict:
    ids = [card_id(c) for c in cards]
    random.shuffle(ids)
    return {"card_order": ids, "next_index": 0, "retry_queue": []}
