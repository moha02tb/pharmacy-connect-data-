"""
Intent classifier for pharmacy-connect chatbot.

Supports two backends:
  - ``spacy``  – lightweight, fast, suitable for production inference
  - ``bert``   – higher accuracy via HuggingFace Transformers fine-tuning

Usage
-----
::

    classifier = IntentClassifier(backend="spacy")
    classifier.train(training_data)           # list of {"text": ..., "intent": ...}
    intent = classifier.predict("What are the side effects of ibuprofen?")
    classifier.save("/path/to/model")
    classifier.load("/path/to/model")
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("spacy", "bert")


class IntentClassifier:
    """
    Multi-backend intent classifier.

    Parameters
    ----------
    backend : str
        ``"spacy"`` (default) or ``"bert"``.
    model_settings : dict, optional
        Override default model hyper-parameters from ``config.py``.
    """

    def __init__(self, backend: str = "spacy", model_settings: Optional[Dict] = None):
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from {SUPPORTED_BACKENDS}."
            )
        self.backend = backend
        self.settings = model_settings or {}
        self._model = None
        self._label_encoder = None
        self._vectorizer = None  # Used by spacy sklearn pipeline
        self._is_trained = False

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, training_data: List[Dict]) -> Dict[str, float]:
        """
        Train the classifier.

        Parameters
        ----------
        training_data : list[dict]
            Each entry must have ``"text"`` and ``"intent"`` keys.

        Returns
        -------
        dict
            Training metrics: ``accuracy``, ``f1``.
        """
        if not training_data:
            raise ValueError("training_data is empty.")

        if self.backend == "spacy":
            return self._train_spacy(training_data)
        return self._train_bert(training_data)

    def predict(self, text: str) -> str:
        """
        Predict the intent label for *text*.

        Returns ``"unknown"`` if the model has not been trained.
        """
        if not self._is_trained:
            logger.warning("Model is not trained yet – returning 'unknown'.")
            return "unknown"

        if self.backend == "spacy":
            return self._predict_spacy(text)
        return self._predict_bert(text)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Return a dict mapping each intent label to its confidence score.
        """
        if not self._is_trained:
            return {}

        if self.backend == "spacy":
            return self._predict_proba_spacy(text)
        return self._predict_proba_bert(text)

    def save(self, path: str) -> None:
        """Persist the trained model to *path*."""
        if not self._is_trained:
            raise RuntimeError("Cannot save: model has not been trained.")

        os.makedirs(path, exist_ok=True)
        if self.backend == "spacy":
            self._save_spacy(path)
        else:
            self._save_bert(path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved model from *path*."""
        if self.backend == "spacy":
            self._load_spacy(path)
        else:
            self._load_bert(path)
        self._is_trained = True
        logger.info("Model loaded from %s", path)

    # ── spaCy backend ─────────────────────────────────────────────────────────

    def _train_spacy(self, training_data: List[Dict]) -> Dict[str, float]:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import LabelEncoder
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for the spaCy backend. "
                "Install it with: pip install scikit-learn"
            ) from exc

        texts = [d["text"] for d in training_data]
        labels = [d["intent"] for d in training_data]

        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)

        test_size = self.settings.get("test_size", 0.2)
        random_state = self.settings.get("random_state", 42)

        x_train, x_test, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, random_state=random_state, stratify=y
        )

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
        }
        logger.info("spaCy backend training metrics: %s", metrics)

        self._model = pipeline
        self._label_encoder = encoder
        self._is_trained = True
        return metrics

    def _predict_spacy(self, text: str) -> str:
        encoded = self._model.predict([text])[0]
        return self._label_encoder.inverse_transform([encoded])[0]

    def _predict_proba_spacy(self, text: str) -> Dict[str, float]:
        proba = self._model.predict_proba([text])[0]
        classes = self._label_encoder.classes_
        return dict(zip(classes, proba.tolist()))

    def _save_spacy(self, path: str) -> None:
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(
                {"pipeline": self._model, "encoder": self._label_encoder}, f
            )

    def _load_spacy(self, path: str) -> None:
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._model = data["pipeline"]
        self._label_encoder = data["encoder"]

    # ── BERT backend ──────────────────────────────────────────────────────────

    def _train_bert(self, training_data: List[Dict]) -> Dict[str, float]:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for the BERT backend. "
                "Install them with: pip install transformers torch scikit-learn"
            ) from exc

        texts = [d["text"] for d in training_data]
        raw_labels = [d["intent"] for d in training_data]

        encoder = LabelEncoder()
        labels = encoder.fit_transform(raw_labels).tolist()
        num_labels = len(encoder.classes_)

        bert_model_name = self.settings.get("bert_model", "bert-base-uncased")
        max_length = self.settings.get("max_length", 128)

        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            bert_model_name, num_labels=num_labels
        )

        # Simple dataset class
        class _IntentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        test_size = self.settings.get("test_size", 0.2)
        x_train, x_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        train_enc = tokenizer(x_train, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        test_enc = tokenizer(x_test, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

        train_dataset = _IntentDataset(train_enc, y_train)
        test_dataset = _IntentDataset(test_enc, y_test)

        training_args = TrainingArguments(
            output_dir=os.path.join(os.getcwd(), "tmp_bert_training"),
            num_train_epochs=self.settings.get("epochs", 3),
            per_device_train_batch_size=self.settings.get("batch_size", 16),
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_steps=10,
            load_best_model_at_end=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        trainer.train()

        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
        }
        logger.info("BERT backend training metrics: %s", metrics)

        self._model = (model, tokenizer)
        self._label_encoder = encoder
        self._is_trained = True
        return metrics

    def _predict_bert(self, text: str) -> str:
        import torch
        import numpy as np

        model, tokenizer = self._model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = int(torch.argmax(logits, dim=1).item())
        return self._label_encoder.inverse_transform([predicted_class])[0]

    def _predict_proba_bert(self, text: str) -> Dict[str, float]:
        import torch
        import torch.nn.functional as F

        model, tokenizer = self._model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        proba = F.softmax(logits, dim=1)[0].tolist()
        return dict(zip(self._label_encoder.classes_, proba))

    def _save_bert(self, path: str) -> None:
        import json as _json

        model, tokenizer = self._model
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        with open(os.path.join(path, "label_encoder.json"), "w") as f:
            _json.dump(self._label_encoder.classes_.tolist(), f)

    def _load_bert(self, path: str) -> None:
        import json as _json
        import numpy as np
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from sklearn.preprocessing import LabelEncoder

        with open(os.path.join(path, "label_encoder.json")) as f:
            classes = _json.load(f)
        encoder = LabelEncoder()
        encoder.classes_ = np.array(classes)
        self._label_encoder = encoder

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        self._model = (model, tokenizer)
