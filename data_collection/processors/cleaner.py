"""
DataCleaner – normalises and deduplicates raw scraped records.
"""

import html
import logging
import re
import unicodedata
from typing import Dict, List

logger = logging.getLogger(__name__)

# Characters to remove (control characters, zero-width spaces, etc.)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class DataCleaner:
    """
    Cleans raw scraped health-data records.

    Operations applied (in order):
      1. Decode HTML entities
      2. Strip residual HTML tags
      3. Remove URLs
      4. Strip control characters
      5. Unicode NFKC normalisation
      6. Collapse whitespace
      7. Title / content length filtering
      8. Deduplication by content fingerprint
    """

    def __init__(self, settings: Dict):
        self.min_length: int = settings.get("min_text_length", 20)
        self.max_length: int = settings.get("max_text_length", 2000)

    # ── Public API ────────────────────────────────────────────────────────────

    def clean(self, records: List[Dict]) -> List[Dict]:
        """
        Clean and deduplicate a list of scraped records.

        Returns the cleaned list (may be shorter than the input if records were
        filtered or merged as duplicates).
        """
        cleaned: List[Dict] = []
        seen_fingerprints: set = set()

        for record in records:
            try:
                cleaned_record = self._clean_record(record)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to clean record from %s: %s", record.get("url"), exc)
                continue

            if cleaned_record is None:
                continue

            fingerprint = self._fingerprint(cleaned_record["content"])
            if fingerprint in seen_fingerprints:
                logger.debug("Duplicate record skipped: %s", cleaned_record.get("url"))
                continue

            seen_fingerprints.add(fingerprint)
            cleaned.append(cleaned_record)

        logger.info("Cleaned %d → %d records (after filtering/dedup)", len(records), len(cleaned))
        return cleaned

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean_record(self, record: Dict) -> Dict | None:
        """Return a cleaned copy of *record*, or ``None`` if it should be discarded."""
        content = self._clean_text(record.get("content", ""))
        title = self._clean_text(record.get("title", ""))

        if len(content) < self.min_length:
            return None

        # Truncate overly long content
        if len(content) > self.max_length:
            content = content[: self.max_length].rsplit(" ", 1)[0] + "…"

        return {
            **record,
            "title": title,
            "content": content,
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Apply all text-cleaning steps and return the result."""
        # 1. Decode HTML entities
        text = html.unescape(text)
        # 2. Strip residual HTML tags
        text = _HTML_TAG_RE.sub(" ", text)
        # 3. Remove URLs
        text = _URL_RE.sub("", text)
        # 4. Remove control characters
        text = _CONTROL_CHAR_RE.sub("", text)
        # 5. Unicode normalisation (NFKC handles ligatures, full-width chars, etc.)
        text = unicodedata.normalize("NFKC", text)
        # 6. Collapse whitespace
        text = _WHITESPACE_RE.sub(" ", text)
        return text.strip()

    @staticmethod
    def _fingerprint(text: str) -> str:
        """Create a simple deduplication fingerprint from *text*."""
        # Lowercase, strip punctuation, join first 200 chars
        normalised = re.sub(r"[^\w\s]", "", text.lower())
        normalised = _WHITESPACE_RE.sub(" ", normalised).strip()
        return normalised[:200]
