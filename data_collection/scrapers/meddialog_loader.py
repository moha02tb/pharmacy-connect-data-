"""
meddialog_loader.py – Load the MedDialog dataset from a local file.

The dataset (OpenMed/MedDialog or any compatible format) can be downloaded
manually from the web and processed locally without any network streaming or
scraping.

Supported local file formats
-----------------------------
* **JSON** – a JSON array of row objects, e.g. ``[{"question": "...",
  "answer": "..."}, ...]``
* **JSONL** – one JSON object per line (JSON Lines / newline-delimited JSON)
* **CSV** – comma-separated values with a header row

Usage
-----
::

    from data_collection.scrapers.meddialog_loader import MedDialogLoader

    # Load from a locally downloaded file
    loader = MedDialogLoader(local_file="data/meddialog.jsonl")
    records = loader.load(max_records=10_000)
"""

import csv
import json
import logging
import os
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
    Load and filter a MedDialog-compatible dataset from a **local file**.

    Download the dataset manually from the web (e.g. from
    https://huggingface.co/datasets/OpenMed/MedDialog) and point this loader
    at the saved file.  No network streaming or scraping is performed.

    Parameters
    ----------
    local_file:
        Path to the locally downloaded dataset file.  Supported formats:

        * ``.json`` – JSON array of row objects
        * ``.jsonl`` / ``.ndjson`` – one JSON object per line
        * ``.csv`` – comma-separated values with a header row

        When ``None`` (and the deprecated ``dataset_name`` parameter is also
        absent), :meth:`load` raises :class:`ValueError`.
    dataset_name:
        *Deprecated.* Kept for backwards compatibility only; ignored when
        ``local_file`` is provided.
    split:
        *Deprecated.* Ignored when ``local_file`` is provided.
    filter_pharmacy:
        When ``True`` (default), keep only conversations that contain at least
        one pharmacy-relevant keyword in the patient utterance.
    cache_dir:
        *Deprecated.* Ignored when loading from a local file.
    use_streaming:
        *Deprecated.* Streaming is never used when loading from a local file.
        This parameter is kept for backwards compatibility only.
    max_retries:
        *Deprecated.* Ignored when loading from a local file.
    """

    def __init__(
        self,
        local_file: Optional[str] = None,
        dataset_name: str = "OpenMed/MedDialog",
        split: str = "train",
        filter_pharmacy: bool = True,
        cache_dir: Optional[str] = None,
        use_streaming: bool = False,
        max_retries: int = 3,
    ) -> None:
        self.local_file = local_file
        self.dataset_name = dataset_name
        self.split = split
        self.filter_pharmacy = filter_pharmacy
        self.cache_dir = cache_dir
        self.use_streaming = use_streaming
        self.max_retries = max(1, max_retries)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, max_records: Optional[int] = None) -> List[Dict]:
        """
        Load conversations from the local dataset file and return structured
        Q&A records.

        Each returned record has the shape::

            {
                "source":   str,           # file name or dataset_name
                "question": str,           # patient utterance
                "answer":   str,           # doctor response
                "category": str,           # always "pharmacy_qa"
                "topics":   list[str],
            }

        Parameters
        ----------
        max_records:
            Maximum number of Q&A pairs to return.  When ``None`` all matching
            pairs are returned.

        Raises
        ------
        ValueError
            When no ``local_file`` path has been configured.
        FileNotFoundError
            When the configured ``local_file`` path does not exist.
        """
        if not self.local_file:
            raise ValueError(
                "No local dataset file configured.  Pass local_file='path/to/dataset.jsonl' "
                "(or set the MEDDIALOG_FILE environment variable) and re-run."
            )

        path = self.local_file
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Dataset file not found: '{path}'.  "
                "Download the dataset manually and provide the correct path."
            )

        source_label = os.path.basename(path)
        logger.info("Loading dataset from local file: %s", path)

        rows = self._read_file(path)
        logger.info("Read %d rows from '%s'.", len(rows), path)

        records: List[Dict] = []
        _first_row_logged = False
        for row in rows:
            pairs = self._extract_qa_pairs(row)
            if not _first_row_logged:
                _first_row_logged = True
                logger.debug(
                    "First dataset row keys: %s  (first pair extracted: %s)",
                    list(row.keys()) if isinstance(row, dict) else type(row).__name__,
                    pairs[0] if pairs else "none",
                )
            for pair in pairs:
                if self.filter_pharmacy and not self._is_pharmacy_relevant(pair["question"]):
                    continue
                records.append(
                    {
                        "source": source_label,
                        "question": pair["question"],
                        "answer": pair["answer"],
                        "category": "pharmacy_qa",
                        "topics": self._extract_topics(pair["question"]),
                    }
                )
                if max_records is not None and len(records) >= max_records:
                    logger.info("Reached max_records limit (%d).", max_records)
                    return records

        logger.info(
            "Loaded %d pharmacy-relevant Q&A pairs from %d rows.",
            len(records),
            len(rows),
        )
        return records

    # ── Local file reader ─────────────────────────────────────────────────────

    def _read_file(self, path: str) -> List[Dict]:
        """
        Read *path* and return a list of row dicts.

        Supported formats are detected by file extension:
        * ``.json``          – JSON array
        * ``.jsonl`` / ``.ndjson`` – newline-delimited JSON (one object per line)
        * ``.csv``           – comma-separated with header row

        Any other extension is tried first as JSONL, then as JSON.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            return self._read_json(path)
        if ext in (".jsonl", ".ndjson"):
            return self._read_jsonl(path)
        if ext == ".csv":
            return self._read_csv(path)
        # Unknown extension – try JSONL first, fall back to JSON
        try:
            rows = self._read_jsonl(path)
            if rows:
                return rows
        except Exception:  # noqa: BLE001
            pass
        return self._read_json(path)

    @staticmethod
    def _read_json(path: str) -> List[Dict]:
        """Read a JSON array file and return the list of objects."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        raise ValueError(
            f"Expected a JSON array in '{path}', got {type(data).__name__}."
        )

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict]:
        """Read a newline-delimited JSON file and return a list of objects."""
        rows: List[Dict] = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON on line %d: %s", lineno, exc)
        return rows

    @staticmethod
    def _read_csv(path: str) -> List[Dict]:
        """Read a CSV file with a header row and return a list of row dicts."""
        rows: List[Dict] = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_qa_pairs(row: Dict) -> List[Dict]:
        """
        Extract Q&A pairs from a single dataset row.

        Supports three MedDialog utterance format variants:

        1. **Dict list** – each element is a dict with ``speaker`` / ``role``
           and ``utterance`` / ``text`` / ``content`` keys.  Patient and doctor
           turns are identified by the ``speaker`` value (e.g. ``"Patient"``,
           ``"Doctor"``).
        2. **Role-prefixed strings** – each element is a string starting with
           a role label such as ``"Patient: ..."`` or ``"Doctor: ..."``.
        3. **Plain alternating strings** – even-indexed elements are the
           patient utterance and odd-indexed elements are the doctor response.

        Multiple field-name aliases are tried in order:
        ``utterances``, ``dialogue``, ``dialog``, ``conversations``,
        ``conversation``, ``messages``, ``turns``.

        When none of those fields are present the method also checks for a
        flat Q&A row (``question`` / ``answer`` keys) and for rows that
        combine a ``description`` field (patient question) with a ``response``
        / ``answer`` field (doctor answer).
        """
        utterances: list = (
            row.get("utterances")
            or row.get("dialogue")
            or row.get("dialog")
            or row.get("conversations")
            or row.get("conversation")
            or row.get("messages")
            or row.get("turns")
            or []
        )

        if not utterances:
            # ── Flat Q&A format: {"question": "...", "answer": "..."} ──────────
            question = str(row.get("question") or row.get("query") or "").strip()
            answer = str(
                row.get("answer") or row.get("response") or row.get("reply") or ""
            ).strip()
            if question and answer:
                return [{"question": question, "answer": answer}]

            # ── Description + answer format ───────────────────────────────────
            description = str(row.get("description") or "").strip()
            answer = str(
                row.get("answer") or row.get("response") or row.get("reply") or ""
            ).strip()
            if description and answer:
                return [{"question": description, "answer": answer}]

            # ── Unknown format – log the row keys on first encounter ──────────
            if row:
                logger.debug(
                    "Row with unrecognised structure (keys: %s) yielded no pairs.",
                    list(row.keys()),
                )
            return []

        pairs: List[Dict] = []

        # ── Format 1: dict-format utterances ─────────────────────────────────
        # (utterances is non-empty at this point; the early return above guards both accesses)
        if isinstance(utterances[0], dict):
            pending_patient: Optional[str] = None
            for utt in utterances:
                speaker = str(utt.get("speaker") or utt.get("role") or "").lower()
                text = str(
                    utt.get("utterance") or utt.get("text") or utt.get("content") or ""
                ).strip()
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
