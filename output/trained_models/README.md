# Trained Models

Saved model artefacts are written here by `scripts/train_models.py`.

Each model is stored in its own subdirectory, e.g.:

```
trained_models/
└── intent_classifier_spacy/
    ├── model.pkl
    └── pipeline_metadata.json
```

This directory is tracked in git (via this README) but model binaries are
excluded via `.gitignore`.
