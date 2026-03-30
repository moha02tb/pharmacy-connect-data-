"""
utils/data_utils.py – Reusable data-manipulation helpers for the pipeline.
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional


def deduplicate_records(records: List[Dict], key: str = "text") -> List[Dict]:
    """
    Return *records* with duplicates removed based on *key*.

    The first occurrence of each unique *key* value is preserved.
    """
    seen: set = set()
    unique: List[Dict] = []
    for record in records:
        fingerprint = record.get(key, "")
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(record)
    return unique


def truncate_text(text: str, max_length: int = 500) -> str:
    """Return *text* truncated to *max_length* characters."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0]


def safe_json_load(path: str, default: Optional[Any] = None) -> Any:
    """
    Load JSON from *path*, returning *default* if the file is missing or
    malformed.
    """
    if not os.path.isfile(path):
        return default if default is not None else []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default if default is not None else []


def safe_json_dump(data: Any, path: str, indent: int = 2) -> None:
    """Write *data* as JSON to *path*, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)


def flatten_conversations(conversations: Iterable[Dict]) -> List[Dict]:
    """
    Flatten a list of multi-turn conversation dicts into individual Q&A pairs.

    Each conversation is expected to have an ``"utterances"`` list where
    alternating entries represent the patient (even indices) and the doctor
    (odd indices).  Returns records of the form::

        {
            "question": str,
            "answer":   str,
            "conv_id":  str | int,
        }
    """
    pairs: List[Dict] = []
    for conv in conversations:
        utterances: List[str] = conv.get("utterances", [])
        conv_id = conv.get("id", "")
        for i in range(0, len(utterances) - 1, 2):
            question = utterances[i].strip()
            answer = utterances[i + 1].strip()
            if question and answer:
                pairs.append(
                    {"question": question, "answer": answer, "conv_id": conv_id}
                )
    return pairs
