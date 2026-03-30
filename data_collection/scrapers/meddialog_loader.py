"""
meddialog_loader.py – Load the OpenMed/MedDialog dataset from Hugging Face.

The OpenMed/MedDialog dataset contains 1.47 M real doctor-patient
conversations, making it an ideal training source for pharmacy intent
classification.

Usage
-----
::

    from data_collection.scrapers.meddialog_loader import MedDialogLoader

    loader = MedDialogLoader()
    records = loader.load(max_records=10_000)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Pharmacy-relevant keywords used to filter conversations
_PHARMACY_KEYWORDS = [
    "medication",
    "medicine",
    "drug",
    "prescription",
    "dosage",
    "dose",
    "side effect",
    "adverse",
    "interaction",
    "pharmacy",
    "pharmacist",
    "pill",
    "tablet",
    "capsule",
    "injection",
    "vaccine",
    "vaccination",
    "supplement",
    "overdose",
    "antibiotic",
    "painkiller",
    "refill",
    "generic",
    "brand",
]


class MedDialogLoader:
    """
    Load and filter the **OpenMed/MedDialog** dataset from Hugging Face.

    Parameters
    ----------
    dataset_name:
        Hugging Face dataset identifier.  Defaults to ``"OpenMed/MedDialog"``.
    split:
        Dataset split to use.  Defaults to ``"train"``.
    filter_pharmacy:
        When ``True`` (default), keep only conversations that contain at least
        one pharmacy-relevant keyword in the patient utterance.
    cache_dir:
        Optional path to a local directory for caching the downloaded dataset.
    """

    def __init__(
        self,
        dataset_name: str = "OpenMed/MedDialog",
        split: str = "train",
        filter_pharmacy: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.filter_pharmacy = filter_pharmacy
        self.cache_dir = cache_dir

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, max_records: Optional[int] = None) -> List[Dict]:
        """
        Load conversations from Hugging Face and return structured Q&A records.

        Each returned record has the shape::

            {
                "source":   "OpenMed/MedDialog",
                "question": str,   # patient utterance
                "answer":   str,   # doctor response
                "category": str,   # always "pharmacy_qa"
                "topics":   list[str],
            }

        Parameters
        ----------
        max_records:
            Maximum number of Q&A pairs to return.  When ``None`` all matching
            pairs are returned.
        """
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required.  "
                "Install it with: pip install datasets>=4.0.0"
            ) from exc

        logger.info("Loading dataset '%s' (split='%s') …", self.dataset_name, self.split)
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        logger.info("Dataset loaded: %d conversations", len(dataset))

        records: List[Dict] = []
        for row in dataset:
            pairs = self._extract_qa_pairs(row)
            for pair in pairs:
                if self.filter_pharmacy and not self._is_pharmacy_relevant(pair["question"]):
                    continue
                records.append(
                    {
                        "source": self.dataset_name,
                        "question": pair["question"],
                        "answer": pair["answer"],
                        "category": "pharmacy_qa",
                        "topics": self._extract_topics(pair["question"]),
                    }
                )
                if max_records is not None and len(records) >= max_records:
                    logger.info("Reached max_records limit (%d).", max_records)
                    return records

        logger.info("Loaded %d pharmacy-relevant Q&A pairs.", len(records))
        return records

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_qa_pairs(row: Dict) -> List[Dict]:
        """
        Extract Q&A pairs from a single dataset row.

        MedDialog stores conversations as a list of utterances.  Even-indexed
        utterances are the patient; odd-indexed are the doctor.
        """
        utterances = row.get("utterances") or []
        # Fallback for datasets that store utterances as 'dialogue'
        if not utterances:
            utterances = row.get("dialogue") or []
        pairs: List[Dict] = []
        for i in range(0, len(utterances) - 1, 2):
            q = str(utterances[i]).strip()
            a = str(utterances[i + 1]).strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
        return pairs

    @staticmethod
    def _is_pharmacy_relevant(text: str) -> bool:
        """Return ``True`` if *text* contains a pharmacy-relevant keyword."""
        lower = text.lower()
        return any(kw in lower for kw in _PHARMACY_KEYWORDS)

    @staticmethod
    def _extract_topics(text: str) -> List[str]:
        """Return the pharmacy keywords present in *text*."""
        lower = text.lower()
        return [kw for kw in _PHARMACY_KEYWORDS if kw in lower]
