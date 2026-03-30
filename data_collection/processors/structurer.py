"""
DataStructurer – converts validated records into the canonical formats used by
the model-training and knowledge-base components.

Output formats
--------------
training_data : list[dict]
    Rows suitable for writing to ``output/training_data.csv``.
    Fields: text, intent, source, category

knowledge_base : list[dict]
    Entries suitable for writing to ``output/knowledge_base.json``.
    Fields: id, title, content, source, url, category, topics
"""

import hashlib
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Simple keyword → intent mapping used to infer intents from content
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "medication_inquiry": [
        "medication",
        "medicine",
        "drug",
        "prescription",
        "pharmaceutical",
    ],
    "drug_interaction_check": [
        "interaction",
        "drug interaction",
        "contraindicated",
        "combine",
    ],
    "side_effects_inquiry": [
        "side effect",
        "adverse",
        "reaction",
        "symptoms after taking",
    ],
    "dosage_inquiry": ["dosage", "dose", "how much", "how to take", "instructions"],
    "vaccination_info": ["vaccine", "vaccination", "immunization", "booster"],
    "first_aid_guidance": ["first aid", "bleeding", "wound", "burn", "cpr", "aed"],
    "emergency_assistance": ["emergency", "call 911", "poison control", "overdose"],
}


class DataStructurer:
    """Transforms validated records into training rows and knowledge-base entries."""

    def __init__(self, settings: Dict | None = None):
        self._settings = settings or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def structure(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Transform *records* into structured outputs.

        Returns a dict with keys ``"training_data"`` and ``"knowledge_base"``.
        """
        training_data: List[Dict] = []
        knowledge_base: List[Dict] = []

        for record in records:
            kb_entry = self._to_knowledge_base_entry(record)
            knowledge_base.append(kb_entry)

            training_row = self._to_training_row(record)
            if training_row:
                training_data.append(training_row)

        logger.info(
            "Structured %d records → %d KB entries, %d training rows",
            len(records),
            len(knowledge_base),
            len(training_data),
        )
        return {"training_data": training_data, "knowledge_base": knowledge_base}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_knowledge_base_entry(self, record: Dict) -> Dict:
        """Convert a validated record into a knowledge-base entry."""
        entry_id = hashlib.md5(  # noqa: S324  (non-security use)
            (record.get("url", "") + record.get("title", "")).encode()
        ).hexdigest()[:12]

        return {
            "id": entry_id,
            "title": record.get("title", ""),
            "content": record.get("content", ""),
            "source": record.get("source", ""),
            "url": record.get("url", ""),
            "category": record.get("category", "general_health"),
            "topics": record.get("topics", []),
        }

    @staticmethod
    def _to_training_row(record: Dict) -> Dict | None:
        """Convert a validated record into a training-data row."""
        content = record.get("content", "")
        if not content:
            return None

        intent = DataStructurer._infer_intent(content, record.get("category", ""))

        return {
            "text": content[:500],  # Truncate for training efficiency
            "intent": intent,
            "source": record.get("source", ""),
            "category": record.get("category", "general_health"),
        }

    @staticmethod
    def _infer_intent(text: str, category: str) -> str:
        """
        Infer a training intent label from content text and category.

        Uses a keyword-matching heuristic; the model-training step will refine
        this with proper NLU.
        """
        lower = text.lower()
        for intent, keywords in _INTENT_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return intent

        # Fall back based on category
        _CATEGORY_INTENT_MAP = {
            "medication": "medication_inquiry",
            "drug_interaction": "drug_interaction_check",
            "side_effects": "side_effects_inquiry",
            "dosage": "dosage_inquiry",
            "vaccination": "vaccination_info",
            "first_aid": "first_aid_guidance",
            "emergency": "emergency_assistance",
        }
        return _CATEGORY_INTENT_MAP.get(category, "general_health_question")
