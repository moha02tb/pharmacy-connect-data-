"""
run_collection.py – Execute the full data-collection pipeline.

Usage
-----
::

    python scripts/run_collection.py [--sources red_cross cdc mayo_clinic]
                                     [--output-dir output/]

This script:
  1. Instantiates each enabled scraper.
  2. Runs them in sequence (respecting per-source rate limits).
  3. Cleans, validates, and structures the scraped records.
  4. Writes ``knowledge_base.json`` and ``training_data.csv`` to the output dir.
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
    PROCESSOR_SETTINGS,
    SCRAPER_SETTINGS,
    TRAINING_DATA_PATH,
)
from data_collection.processors import DataCleaner, DataStructurer, DataValidator
from data_collection.scrapers import CDCScraper, MayoClinicScraper, RedCrossScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SCRAPER_MAP = {
    "red_cross": RedCrossScraper,
    "cdc": CDCScraper,
    "mayo_clinic": MayoClinicScraper,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pharmacy Connect – data collection")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(_SCRAPER_MAP.keys()),
        default=list(_SCRAPER_MAP.keys()),
        help="Data sources to scrape (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for output files (default: output/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scrapers but do not write output files",
    )
    return parser.parse_args()


def run_collection(sources: list, output_dir: str, dry_run: bool = False) -> dict:
    """
    Execute the data-collection pipeline.

    Returns a summary dict with record counts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Scraping ──────────────────────────────────────────────────────────────
    all_raw_records = []
    for source_key in sources:
        source_config = DATA_SOURCES.get(source_key)
        if not source_config:
            logger.warning("Unknown source '%s' – skipping.", source_key)
            continue

        scraper_cls = _SCRAPER_MAP[source_key]
        scraper = scraper_cls(config=source_config)
        records = scraper.scrape()
        all_raw_records.extend(records)
        logger.info("Source '%s': %d raw records", source_key, len(records))

    logger.info("Total raw records: %d", len(all_raw_records))

    # ── Processing ────────────────────────────────────────────────────────────
    cleaner = DataCleaner(settings=PROCESSOR_SETTINGS)
    cleaned = cleaner.clean(all_raw_records)

    validator = DataValidator(settings=PROCESSOR_SETTINGS)
    validated = validator.validate(cleaned)

    structurer = DataStructurer(settings=PROCESSOR_SETTINGS)
    structured = structurer.structure(validated)

    knowledge_base = structured["knowledge_base"]
    training_data = structured["training_data"]

    logger.info(
        "Pipeline complete: %d KB entries, %d training rows",
        len(knowledge_base),
        len(training_data),
    )

    # ── Output ────────────────────────────────────────────────────────────────
    if not dry_run:
        kb_path = os.path.join(output_dir, os.path.basename(KNOWLEDGE_BASE_PATH))
        _write_knowledge_base(knowledge_base, kb_path)

        td_path = os.path.join(output_dir, os.path.basename(TRAINING_DATA_PATH))
        _write_training_data(training_data, td_path)
    else:
        logger.info("Dry run – output files not written.")

    return {
        "raw_records": len(all_raw_records),
        "cleaned_records": len(cleaned),
        "validated_records": len(validated),
        "knowledge_base_entries": len(knowledge_base),
        "training_rows": len(training_data),
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

    # Merge: preserve existing seed entries, append new ones
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
        sources=args.sources,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    print("\nCollection summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
