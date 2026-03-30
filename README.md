# pharmacy-connect-data

Data pipeline and NLU model training repository for the **PharmacyConnect** chatbot.

> **v2** – Legacy web scrapers (Red Cross, CDC, Mayo Clinic) replaced by the
> **OpenMed/MedDialog** Hugging Face dataset (1.47 M doctor-patient conversations).
> Training data increased from **53 rows → 65 000+ rows**.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Load MedDialog & build training data (65 000+ Q&A pairs)
python scripts/run_collection.py

# 2. Train the intent classifier
python scripts/train_models.py --backend spacy

# 3. Export artefacts for the main app
python scripts/export_for_app.py --output-dir export/
```

See **[docs/QUICK_START.md](docs/QUICK_START.md)** for a detailed walkthrough.

## Repository Structure

```
pharmacy-connect-data/
├── data_collection/
│   ├── scrapers/          MedDialogLoader, MedDialogKBBuilder, DrugBankLoader
│   ├── processors/        IntentMapper, MedDialogProcessor, cleaner, validator, structurer
│   ├── config.py          All configurable settings
│   └── meddialog_config.yaml  Dataset-specific configuration
├── model_training/
│   ├── intent_classifier.py    spaCy / BERT
│   ├── response_generator.py   Template + NLU retrieval
│   ├── data_augmentation.py    Synthetic data generation
│   └── training_pipeline.py   Orchestration
├── utils/
│   ├── logger.py          Consistent logging
│   └── data_utils.py      Data manipulation helpers
├── output/
│   ├── trained_models/         Saved model artefacts
│   ├── knowledge_base.json     Structured health content
│   ├── training_data.csv       Labelled intent examples
│   └── embeddings/             Pre-computed vectors
├── scripts/
│   ├── run_collection.py       Execute data collection (MedDialog pipeline)
│   ├── train_models.py         Train NLU models
│   └── export_for_app.py       Package for main app
├── tests/
│   └── test_meddialog_loader.py  Unit tests
├── notebooks/
│   └── meddialog_exploration.ipynb  Interactive exploration
└── docs/
    ├── DATA_PIPELINE.md        Full technical documentation
    └── QUICK_START.md          Quick start guide
```

## Data Sources

| Source | Type | Records | Status |
|--------|------|---------|--------|
| **OpenMed/MedDialog** | Hugging Face dataset | 1.47 M conversations | ✅ Active |
| **DrugBank Open Data** | Local JSON/CSV | 13 000+ drugs | ✅ Optional |
| Red Cross | Web scraper | — | ❌ Removed (403 Forbidden) |
| CDC | Web scraper | — | ❌ Removed (404 broken) |
| Mayo Clinic | Web scraper | — | ❌ Removed (rate-limited) |

See **[docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)** for the full technical reference.