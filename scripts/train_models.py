"""
train_models.py – Train pharmacy-connect NLU models.

Usage
-----
::

    python scripts/train_models.py [--backend spacy|bert]
                                   [--training-data output/training_data.csv]
                                   [--knowledge-base output/knowledge_base.json]
                                   [--models-dir output/trained_models/]

Steps
-----
1. Load and validate training data from CSV.
2. Run data augmentation to expand the dataset.
3. Train the intent classifier with the selected backend.
4. Save the model artefacts to ``output/trained_models/``.
5. Print evaluation metrics.
"""

import argparse
import json
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_collection.config import (
    KNOWLEDGE_BASE_PATH,
    TRAINED_MODELS_DIR,
    TRAINING_DATA_PATH,
)
from model_training.training_pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pharmacy Connect – model training")
    parser.add_argument(
        "--backend",
        choices=["spacy", "bert"],
        default="spacy",
        help="Training backend to use (default: spacy)",
    )
    parser.add_argument(
        "--training-data",
        default=TRAINING_DATA_PATH,
        help="Path to training_data.csv",
    )
    parser.add_argument(
        "--knowledge-base",
        default=KNOWLEDGE_BASE_PATH,
        help="Path to knowledge_base.json",
    )
    parser.add_argument(
        "--models-dir",
        default=TRAINED_MODELS_DIR,
        help="Directory to save trained models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = TrainingPipeline(
        backend=args.backend,
        training_data_path=args.training_data,
        knowledge_base_path=args.knowledge_base,
        models_dir=args.models_dir,
    )

    results = pipeline.run()

    print("\n=== Training complete ===")
    print(f"  Training samples (original): {results['training_samples']}")
    print(f"  Training samples (augmented): {results['augmented_samples']}")
    accuracy = results['metrics'].get('accuracy')
    f1 = results['metrics'].get('f1')
    print(f"  Accuracy: {accuracy:.4f}" if isinstance(accuracy, float) else f"  Accuracy: {accuracy}")
    print(f"  F1 score: {f1:.4f}" if isinstance(f1, float) else f"  F1 score: {f1}")
    print(f"  Model saved to: {results['model_path']}")


if __name__ == "__main__":
    main()
