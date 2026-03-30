"""
drugbank_loader.py – Enrich the knowledge base with DrugBank Open Data.

DrugBank Open Data provides metadata for 13 000+ drugs including indications,
mechanisms of action, and side-effect profiles.  The free tier is available as
a CSV/JSON download from https://go.drugbank.com/releases/latest.

This loader accepts a locally-downloaded JSON or CSV file and converts each
drug entry into a knowledge-base record.

Usage
-----
::

    from data_collection.scrapers.drugbank_loader import DrugBankLoader

    loader = DrugBankLoader(file_path="data/drugbank_open.csv")
    records = loader.load()
"""

import csv
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DrugBankLoader:
    """
    Load DrugBank Open Data from a local file and produce knowledge-base entries.

    Supported file formats: ``.json`` and ``.csv``.

    Parameters
    ----------
    file_path:
        Path to the downloaded DrugBank file.
    max_records:
        Maximum number of records to load.  ``None`` means no limit.
    """

    def __init__(
        self,
        file_path: str,
        max_records: Optional[int] = None,
    ) -> None:
        self.file_path = file_path
        self.max_records = max_records

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> List[Dict]:
        """
        Load drug records from *file_path* and return knowledge-base entries.

        Each entry has the shape::

            {
                "id":       str,
                "title":    str,   # drug name
                "content":  str,   # indication + description
                "source":   "DrugBank",
                "url":      str,
                "category": "medication",
                "topics":   list[str],
            }
        """
        if not os.path.isfile(self.file_path):
            logger.warning("DrugBank file not found: %s – skipping.", self.file_path)
            return []

        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == ".json":
            raw_records = self._load_json()
        elif ext == ".csv":
            raw_records = self._load_csv()
        else:
            raise ValueError(f"Unsupported DrugBank file format: '{ext}'")

        logger.info("Loaded %d raw DrugBank records.", len(raw_records))
        entries = [self._to_kb_entry(r) for r in raw_records]
        if self.max_records is not None:
            entries = entries[: self.max_records]
        logger.info("Returning %d DrugBank KB entries.", len(entries))
        return entries

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_json(self) -> List[Dict]:
        with open(self.file_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        # Some exports wrap entries under a "drugs" key
        return data.get("drugs", [])

    def _load_csv(self) -> List[Dict]:
        records: List[Dict] = []
        with open(self.file_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                records.append(dict(row))
        return records

    @staticmethod
    def _to_kb_entry(record: Dict) -> Dict:
        name = record.get("name") or record.get("drug_name") or "Unknown Drug"
        indication = record.get("indication") or record.get("description") or ""
        url = record.get("drugbank_id") or ""
        if url:
            url = f"https://go.drugbank.com/drugs/{url}"

        content_parts = []
        if indication:
            content_parts.append(indication)
        for field in ("mechanism_of_action", "side_effects", "dosage"):
            value = record.get(field, "")
            if value:
                content_parts.append(f"{field.replace('_', ' ').title()}: {value}")

        content = " ".join(content_parts)[:1000]

        entry_id = hashlib.md5(  # noqa: S324  (non-security use)
            name.encode("utf-8", errors="replace")
        ).hexdigest()[:12]

        topics = ["medication", "drug"]
        for field in ("drug_interactions", "side_effects"):
            if record.get(field):
                topics.append(field.replace("_", " "))

        return {
            "id": entry_id,
            "title": name,
            "content": content,
            "source": "DrugBank",
            "url": url,
            "category": "medication",
            "topics": topics,
        }
