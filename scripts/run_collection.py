"""
run_collection.py – Execute the MedDialog data-collection pipeline.

Usage
-----
::

    python scripts/run_collection.py --dataset-file path/to/meddialog.jsonl
                                     [--max-records 65000]
                                     [--output-dir output/]
                                     [--drugbank-file path/to/drugbank.json]

How to get the dataset
-----------------------
1. Visit https://huggingface.co/datasets/OpenMed/MedDialog
2. Download the dataset file(s) you need (JSON, JSONL, or CSV).
3. Pass the local path via ``--dataset-file`` (or set ``MEDDIALOG_FILE``).

Pipeline steps
--------------
1. **Load** – Read the locally downloaded MedDialog dataset file and filter
   for pharmacy-relevant conversations.
2. **Process** – Map Q&A pairs to 13 pharmacy intent labels and build
   structured training rows + knowledge-base entries.
3. **Export** – Write ``knowledge_base.json`` and ``training_data.csv``.

Optional: if ``--drugbank-file`` is provided (or the ``DRUGBANK_FILE``
environment variable is set), DrugBank drug metadata is merged into the
knowledge base.
"""

import argparse
import csv
import json
import logging
import os
import sys

# Ensure the project root is importable regardless of where the script is run from
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_collection.config import (
    DATA_SOURCES,
    KNOWLEDGE_BASE_PATH,
    OUTPUT_DIR,
    TRAINING_DATA_PATH,
)
from data_collection.scrapers import DrugBankLoader, MedDialogKBBuilder, MedDialogLoader
from data_collection.processors import IntentMapper, MedDialogProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pharmacy Connect – MedDialog data collection")
    parser.add_argument(
        "--dataset-file",
        default=DATA_SOURCES["meddialog"].get("local_file", ""),
        help=(
            "Path to the locally downloaded MedDialog dataset file "
            "(JSON, JSONL, or CSV).  Can also be set via the MEDDIALOG_FILE "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=DATA_SOURCES["meddialog"].get("max_records", 65_000),
        help="Maximum Q&A pairs to extract from the dataset (default: 65 000)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for output files (default: output/)",
    )
    parser.add_argument(
        "--drugbank-file",
        default=DATA_SOURCES["drugbank"].get("file_path", ""),
        help="Path to a local DrugBank JSON/CSV file (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline but do not write output files",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable pharmacy-keyword filtering (load all conversations)",
    )
    return parser.parse_args()


def run_collection(
    dataset_file: str,
    max_records: int,
    output_dir: str,
    drugbank_file: str = "",
    dry_run: bool = False,
    filter_pharmacy: bool = True,
) -> dict:
    """
    Execute the 3-step MedDialog data-collection pipeline.

    Returns a summary dict with record counts.
    """
    os.makedirs(output_dir, exist_ok=True)

    meddialog_cfg = DATA_SOURCES["meddialog"]

    # ── Step 1: Load MedDialog ────────────────────────────────────────────────
    logger.info("Step 1/3 – Loading MedDialog dataset from local file …")
    loader = MedDialogLoader(
        local_file=dataset_file or meddialog_cfg.get("local_file", ""),
        filter_pharmacy=filter_pharmacy and meddialog_cfg.get("filter_pharmacy", True),
    )
    qa_pairs = loader.load(max_records=max_records)
    logger.info("Loaded %d Q&A pairs from MedDialog.", len(qa_pairs))

    # ── Step 2: Process – intent mapping + training rows + KB ─────────────────
    logger.info("Step 2/3 – Mapping intents and building training data …")
    mapper = IntentMapper(
        confidence_threshold=MEDDIALOG_SETTINGS.get("confidence_threshold", 0.3)
    )
    qa_pairs = mapper.map(qa_pairs)

    processor = MedDialogProcessor(
        max_text_length=MEDDIALOG_SETTINGS.get("max_text_length", 500),
        min_text_length=MEDDIALOG_SETTINGS.get("min_text_length", 10),
        include_answer=MEDDIALOG_SETTINGS.get("include_answer_in_text", False),
    )
    training_rows = processor.process(qa_pairs)

    kb_builder = MedDialogKBBuilder()
    knowledge_base = kb_builder.build(qa_pairs)

    # ── Optional: DrugBank enrichment ─────────────────────────────────────────
    drugbank_entries: list = []
    if drugbank_file:
        logger.info("Loading DrugBank data from '%s' …", drugbank_file)
        db_loader = DrugBankLoader(file_path=drugbank_file)
        drugbank_entries = db_loader.load()
        knowledge_base.extend(drugbank_entries)
        logger.info("Added %d DrugBank entries to knowledge base.", len(drugbank_entries))

    logger.info(
        "Step 2/3 complete: %d training rows, %d KB entries",
        len(training_rows),
        len(knowledge_base),
    )

    # ── Step 3: Export ────────────────────────────────────────────────────────
    logger.info("Step 3/3 – Writing output files …")
    if not dry_run:
        kb_path = os.path.join(output_dir, os.path.basename(KNOWLEDGE_BASE_PATH))
        _write_knowledge_base(knowledge_base, kb_path)

        td_path = os.path.join(output_dir, os.path.basename(TRAINING_DATA_PATH))
        _write_training_data(training_rows, td_path)
    else:
        logger.info("Dry run – output files not written.")

    return {
        "qa_pairs_loaded": len(qa_pairs),
        "training_rows": len(training_rows),
        "knowledge_base_entries": len(knowledge_base),
        "drugbank_entries": len(drugbank_entries),
    }


def _write_knowledge_base(entries: list, path: str) -> None:
    """Merge new entries with any existing knowledge base and write JSON."""
    existing: list = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Existing KB at %s is malformed – overwriting.", path)

    existing_ids = {e["id"] for e in existing}
    new_entries = [e for e in entries if e["id"] not in existing_ids]
    merged = existing + new_entries

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    logger.info("Knowledge base written: %d total entries → %s", len(merged), path)


def _write_training_data(rows: list, path: str) -> None:
    """Append new training rows to the CSV (or create it)."""
    fieldnames = ["text", "intent", "source", "category"]
    write_header = not os.path.isfile(path)

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logger.info("Training data written: %d rows → %s", len(rows), path)


if __name__ == "__main__":
    args = parse_args()
    summary = run_collection(
        dataset_file=args.dataset_file,
        max_records=args.max_records,
        output_dir=args.output_dir,
        drugbank_file=args.drugbank_file,
        dry_run=args.dry_run,
        filter_pharmacy=not args.no_filter,
    )
    print("\nCollection summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
