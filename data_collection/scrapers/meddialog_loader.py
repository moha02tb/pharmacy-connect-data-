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

# Speaker-role substrings used to identify patient vs. doctor turns in
# dict-format utterances (matched case-insensitively via ``in``).
_PATIENT_ROLES = ("patient", "user", "customer")
_DOCTOR_ROLES = ("doctor", "physician", "assistant")

# Role-label prefixes used to identify patient vs. doctor turns in
# role-prefixed string utterances (matched case-insensitively via
# ``startswith``).  Abbreviations ``p:`` and ``d:`` are included for
# datasets that use terse role labels.
_PATIENT_PREFIXES: tuple = tuple(r + ":" for r in _PATIENT_ROLES) + ("p:",)
_DOCTOR_PREFIXES: tuple = tuple(r + ":" for r in _DOCTOR_ROLES) + ("d:",)
_ALL_ROLE_PREFIXES: tuple = _PATIENT_PREFIXES + _DOCTOR_PREFIXES


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

        Supports three MedDialog utterance format variants:

        1. **Dict list** – each element is a dict with ``speaker`` / ``role``
           and ``utterance`` / ``text`` keys.  Patient and doctor turns are
           identified by the ``speaker`` value (e.g. ``"Patient"``,
           ``"Doctor"``).
        2. **Role-prefixed strings** – each element is a string starting with
           a role label such as ``"Patient: ..."`` or ``"Doctor: ..."``.
        3. **Plain alternating strings** – even-indexed elements are the
           patient utterance and odd-indexed elements are the doctor response.

        Multiple field-name aliases are tried in order:
        ``utterances``, ``dialogue``, ``dialog``, ``conversations``.
        """
        utterances: list = (
            row.get("utterances")
            or row.get("dialogue")
            or row.get("dialog")
            or row.get("conversations")
            or []
        )

        if not utterances:
            return []

        pairs: List[Dict] = []

        # ── Format 1: dict-format utterances ─────────────────────────────────
        # (utterances is non-empty at this point; the early return above guards both accesses)
        if isinstance(utterances[0], dict):
            pending_patient: Optional[str] = None
            for utt in utterances:
                speaker = str(utt.get("speaker") or utt.get("role") or "").lower()
                text = str(utt.get("utterance") or utt.get("text") or "").strip()
                if not text:
                    continue
                if any(s in speaker for s in _PATIENT_ROLES):
                    pending_patient = text
                elif any(s in speaker for s in _DOCTOR_ROLES):
                    if pending_patient:
                        pairs.append({"question": pending_patient, "answer": text})
                        pending_patient = None
            return pairs

        # ── Format 2 & 3: string utterances ──────────────────────────────────
        first = str(utterances[0]).lower()
        if any(first.startswith(p) for p in _ALL_ROLE_PREFIXES):
            # Role-prefixed strings
            pending_patient = None
            for utt in utterances:
                raw = str(utt)
                lower = raw.lower()
                is_patient = any(lower.startswith(p) for p in _PATIENT_PREFIXES)
                is_doctor = any(lower.startswith(p) for p in _DOCTOR_PREFIXES)
                # Strip the role prefix
                text = raw
                for prefix in _ALL_ROLE_PREFIXES:
                    if lower.startswith(prefix):
                        text = raw[len(prefix):].strip()
                        break
                if not text:
                    continue
                if is_patient:
                    pending_patient = text
                elif is_doctor and pending_patient:
                    pairs.append({"question": pending_patient, "answer": text})
                    pending_patient = None
            return pairs

        # Plain alternating string list (even = patient, odd = doctor)
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
