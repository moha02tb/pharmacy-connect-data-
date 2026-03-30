"""
DataValidator – validates cleaned records against a defined schema and
enforces field-level quality rules.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("source", "url", "title", "content", "category")
_VALID_URL_RE = re.compile(r"^https?://")


class DataValidator:
    """
    Validates cleaned health-data records.

    Checks performed:
      - All required fields are present and non-empty
      - URL format is valid
      - Content length is within acceptable bounds
      - Category is in the allowed set
      - Language is detected as English (basic heuristic)
    """

    def __init__(self, settings: Dict):
        self.min_length: int = settings.get("min_text_length", 20)
        self.max_length: int = settings.get("max_text_length", 2000)
        self.allowed_categories: List[str] = settings.get(
            "allowed_categories",
            [
                "medication",
                "pharmacy",
                "first_aid",
                "vaccination",
                "drug_interaction",
                "dosage",
                "side_effects",
                "emergency",
                "general_health",
            ],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, records: List[Dict]) -> List[Dict]:
        """
        Validate a list of records.

        Returns only records that pass all validation rules.
        """
        valid: List[Dict] = []
        for record in records:
            passed, reasons = self._validate_record(record)
            if passed:
                valid.append(record)
            else:
                logger.debug(
                    "Record from %s failed validation: %s",
                    record.get("url", "unknown"),
                    "; ".join(reasons),
                )

        logger.info(
            "Validated %d → %d records (%d failed)",
            len(records),
            len(valid),
            len(records) - len(valid),
        )
        return valid

    def validate_record(self, record: Dict) -> Tuple[bool, List[str]]:
        """Public single-record validation (convenience wrapper)."""
        return self._validate_record(record)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_record(self, record: Dict) -> Tuple[bool, List[str]]:
        """Return ``(passed, list_of_failure_reasons)`` for a single record."""
        reasons: List[str] = []

        # 1. Required fields
        for field in _REQUIRED_FIELDS:
            if not record.get(field):
                reasons.append(f"Missing or empty field: '{field}'")

        # 2. URL format
        url = record.get("url", "")
        if url and not _VALID_URL_RE.match(url):
            reasons.append(f"Invalid URL format: '{url}'")

        # 3. Content length
        content = record.get("content", "")
        if content:
            if len(content) < self.min_length:
                reasons.append(
                    f"Content too short ({len(content)} < {self.min_length} chars)"
                )
            if len(content) > self.max_length:
                reasons.append(
                    f"Content too long ({len(content)} > {self.max_length} chars)"
                )

        # 4. Category allowlist
        category = record.get("category", "")
        if category and category not in self.allowed_categories:
            reasons.append(f"Unknown category: '{category}'")

        # 5. Basic English heuristic (ASCII ratio)
        if content and self._non_ascii_ratio(content) > 0.3:
            reasons.append("Content appears to be non-English")

        return (len(reasons) == 0), reasons

    @staticmethod
    def _non_ascii_ratio(text: str) -> float:
        """Return the proportion of non-ASCII characters in *text*."""
        if not text:
            return 0.0
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        return non_ascii / len(text)
