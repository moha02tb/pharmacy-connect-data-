"""
meddialog_kb_builder.py – Build a structured knowledge base from MedDialog Q&A pairs.

Takes the output of :class:`MedDialogLoader` and produces knowledge-base entries
suitable for writing to ``output/knowledge_base.json``.
"""

import hashlib
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class MedDialogKBBuilder:
    """
    Convert MedDialog Q&A pairs into knowledge-base entries.

    Parameters
    ----------
    max_answer_length:
        Maximum character length for the ``content`` field in each KB entry.
        Defaults to ``1000``.
    """

    def __init__(self, max_answer_length: int = 1000) -> None:
        self.max_answer_length = max_answer_length

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Convert *qa_pairs* (from :meth:`MedDialogLoader.load`) into KB entries.

        Each entry has the shape::

            {
                "id":       str,   # 12-char MD5 of question
                "title":    str,   # first 120 chars of question
                "content":  str,   # doctor answer (truncated)
                "source":   str,
                "url":      str,   # empty – dataset has no URLs
                "category": str,
                "topics":   list[str],
            }
        """
        entries: List[Dict] = []
        seen_ids: set = set()

        for pair in qa_pairs:
            entry = self._to_kb_entry(pair)
            if entry["id"] not in seen_ids:
                seen_ids.add(entry["id"])
                entries.append(entry)

        logger.info("Built %d knowledge-base entries from %d pairs.", len(entries), len(qa_pairs))
        return entries

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_kb_entry(self, pair: Dict) -> Dict:
        question = pair.get("question", "")
        answer = pair.get("answer", "")

        entry_id = hashlib.md5(  # noqa: S324  (non-security use)
            question.encode("utf-8", errors="replace")
        ).hexdigest()[:12]

        content = answer[: self.max_answer_length]

        return {
            "id": entry_id,
            "title": question[:120],
            "content": content,
            "source": pair.get("source", "OpenMed/MedDialog"),
            "url": "",
            "category": pair.get("category", "pharmacy_qa"),
            "topics": pair.get("topics", []),
        }
