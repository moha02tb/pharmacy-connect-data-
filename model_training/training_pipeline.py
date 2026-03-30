"""
Training pipeline – orchestrates data loading, augmentation, model training,
evaluation, and artefact saving.

Usage (from scripts/train_models.py):
::

    from model_training.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline()
    results = pipeline.run()
"""

import csv
import json
import logging
import os
import sys
from typing import Dict, List, Optional

# Ensure the project root is on sys.path when run directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_collection.config import (
    KNOWLEDGE_BASE_PATH,
    MODEL_SETTINGS,
    TRAINED_MODELS_DIR,
    TRAINING_DATA_PATH,
)
from model_training.data_augmentation import DataAugmenter
from model_training.intent_classifier import IntentClassifier
from model_training.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end model-training pipeline.

    Steps
    -----
    1. Load training data from ``output/training_data.csv``.
    2. Augment with synthetic samples.
    3. Train the intent classifier.
    4. Instantiate and validate the response generator.
    5. Save all artefacts to ``output/trained_models/``.
    """

    def __init__(
        self,
        backend: str = "spacy",
        training_data_path: Optional[str] = None,
        knowledge_base_path: Optional[str] = None,
        models_dir: Optional[str] = None,
    ):
        self.backend = backend
        self.training_data_path = training_data_path or TRAINING_DATA_PATH
        self.knowledge_base_path = knowledge_base_path or KNOWLEDGE_BASE_PATH
        self.models_dir = models_dir or TRAINED_MODELS_DIR

        self._intent_settings = MODEL_SETTINGS.get("intent_classifier", {})
        self._response_settings = MODEL_SETTINGS.get("response_generator", {})
        self._augmentation_settings = MODEL_SETTINGS.get("data_augmentation", {})

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Execute the full training pipeline.

        Returns
        -------
        dict
            Pipeline results: ``training_samples``, ``augmented_samples``,
            ``metrics``, ``model_path``.
        """
        logger.info("=== Training pipeline started (backend=%s) ===", self.backend)

        # Step 1: Load data
        training_data = self._load_training_data()
        if not training_data:
            raise RuntimeError(
                f"No training data found at {self.training_data_path}. "
                "Run data collection first (scripts/run_collection.py)."
            )
        logger.info("Loaded %d training samples.", len(training_data))

        # Step 2: Augment
        augmenter = DataAugmenter(settings=self._augmentation_settings)
        augmented_data = augmenter.augment(training_data)
        logger.info("Augmented to %d samples.", len(augmented_data))

        # Step 3: Train classifier
        classifier = IntentClassifier(
            backend=self.backend, model_settings=self._intent_settings
        )
        metrics = classifier.train(augmented_data)
        logger.info("Classifier training metrics: %s", metrics)

        # Step 4: Instantiate response generator (validates KB)
        generator = ResponseGenerator(
            knowledge_base_path=self.knowledge_base_path,
            settings=self._response_settings,
        )

        # Step 5: Save artefacts
        model_path = os.path.join(self.models_dir, f"intent_classifier_{self.backend}")
        os.makedirs(model_path, exist_ok=True)
        classifier.save(model_path)

        self._save_pipeline_metadata(
            model_path=model_path,
            metrics=metrics,
            n_train=len(training_data),
            n_augmented=len(augmented_data),
        )

        logger.info("=== Training pipeline complete. Model saved to %s ===", model_path)

        return {
            "training_samples": len(training_data),
            "augmented_samples": len(augmented_data),
            "metrics": metrics,
            "model_path": model_path,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_training_data(self) -> List[Dict]:
        """Load training rows from the CSV file."""
        if not os.path.isfile(self.training_data_path):
            return []

        rows: List[Dict] = []
        with open(self.training_data_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("text") and row.get("intent"):
                    rows.append({"text": row["text"], "intent": row["intent"]})

        return rows

    @staticmethod
    def _save_pipeline_metadata(
        model_path: str,
        metrics: Dict,
        n_train: int,
        n_augmented: int,
    ) -> None:
        """Write a JSON metadata file alongside the saved model."""
        meta = {
            "training_samples": n_train,
            "augmented_samples": n_augmented,
            "metrics": metrics,
        }
        meta_path = os.path.join(model_path, "pipeline_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Pipeline metadata saved to %s", meta_path)
