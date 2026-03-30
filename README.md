# pharmacy-connect-data

Data pipeline and NLU model training repository for the **PharmacyConnect** chatbot.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Collect health data from Red Cross, CDC, and Mayo Clinic
python scripts/run_collection.py

# 2. Train the intent classifier
python scripts/train_models.py --backend spacy

# 3. Export artefacts for the main app
python scripts/export_for_app.py --output-dir export/
```

## Repository Structure

```
pharmacy-connect-data/
├── data_collection/
│   ├── scrapers/          Red Cross, CDC, Mayo Clinic
│   ├── processors/        Clean, validate, structure
│   └── config.py          All configurable settings
├── model_training/
│   ├── intent_classifier.py    spaCy / BERT
│   ├── response_generator.py   Template + NLU retrieval
│   ├── data_augmentation.py    Synthetic data generation
│   └── training_pipeline.py   Orchestration
├── output/
│   ├── trained_models/         Saved model artefacts
│   ├── knowledge_base.json     Structured health content
│   ├── training_data.csv       Labelled intent examples
│   └── embeddings/             Pre-computed vectors
├── scripts/
│   ├── run_collection.py       Execute data collection
│   ├── train_models.py         Train NLU models
│   └── export_for_app.py       Package for main app
└── docs/
    └── DATA_PIPELINE.md        Full technical documentation
```

See **[docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)** for the full technical reference.