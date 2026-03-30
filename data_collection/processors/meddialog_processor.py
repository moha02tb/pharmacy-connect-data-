"""
meddialog_processor.py – Convert MedDialog Q&A pairs into training-data rows.

Takes the output of :class:`MedDialogLoader` (after intent mapping) and
produces rows suitable for writing to ``output/training_data.csv``.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum length for the ``text`` field in training rows
_DEFAULT_MAX_TEXT_LENGTH = 500


class MedDialogProcessor:
    """
    Process MedDialog Q&A pairs into structured training-data rows.

    Parameters
    ----------
    max_text_length:
        Maximum character length for the ``text`` field.  Defaults to 500.
    min_text_length:
        Minimum character length; shorter questions are discarded.
        Defaults to 10.
    include_answer:
        When ``True`` the doctor's answer is appended to the question text,
        separated by a newline, to give the model richer context.  Defaults to
        ``False`` to keep training examples concise.
    """

    def __init__(
        self,
        max_text_length: int = _DEFAULT_MAX_TEXT_LENGTH,
        min_text_length: int = 10,
        include_answer: bool = False,
    ) -> None:
        self.max_text_length = max_text_length
        self.min_text_length = min_text_length
        self.include_answer = include_answer

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Convert *qa_pairs* into training-data rows.

        Each input dict should contain at least ``"question"``, ``"intent"``,
        and ``"source"`` keys (i.e. the output of :meth:`IntentMapper.map`).

        Returns a list of dicts with the shape::

            {
                "text":     str,
                "intent":   str,
                "source":   str,
                "category": str,
            }
        """
        rows: List[Dict] = []
        skipped = 0

        for pair in qa_pairs:
            row = self._to_training_row(pair)
            if row is None:
                skipped += 1
                continue
            rows.append(row)

        logger.info(
            "Processed %d Q&A pairs → %d training rows (%d skipped).",
            len(qa_pairs),
            len(rows),
            skipped,
        )
        return rows

    def filter_intents(
        self,
        rows: List[Dict],
        allowed_intents: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Return only rows whose intent is in *allowed_intents*."""
        if allowed_intents is None:
            return rows
        return [r for r in rows if r.get("intent") in allowed_intents]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_training_row(self, pair: Dict) -> Optional[Dict]:
        question = pair.get("question", "").strip()
        answer = pair.get("answer", "").strip()

        if len(question) < self.min_text_length:
            return None

        if self.include_answer and answer:
            text = f"{question}\n{answer}"
        else:
            text = question

        # Truncate at word boundary
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length].rsplit(" ", 1)[0]

        return {
            "text": text,
            "intent": pair.get("intent", "general_health_question"),
            "source": pair.get("source", "OpenMed/MedDialog"),
            "category": pair.get("category", "pharmacy_qa"),
        }
