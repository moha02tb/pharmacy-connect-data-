# Data Pipeline – Technical Reference

This document describes the end-to-end data pipeline for the
**pharmacy-connect-data** repository, covering data collection, processing,
model training, and artefact export.

---

## Overview

```
Data Sources                Processing               Model Training        Export
─────────────────────────────────────────────────────────────────────────────────
Red Cross  ──┐              ┌─────────────┐          ┌──────────────┐      ┌──────────┐
CDC        ──┼──► Scraper ──► Cleaner     ──► Struct ──► IntentCls   ──────► manifest  │
Mayo Clinic ─┘              │ Validator   │  urer    │ ResponseGen  │      │ KB copy  │
                             └─────────────┘          │ DataAugment  │      │ Embeds   │
                                                       └──────────────┘      └──────────┘
```

---

## Directory Structure

```
pharmacy-connect-data/
├── data_collection/
│   ├── scrapers/
│   │   ├── base_scraper.py        Abstract base class
│   │   ├── red_cross_scraper.py   American Red Cross
│   │   ├── cdc_scraper.py         CDC
│   │   └── mayo_clinic_scraper.py Mayo Clinic
│   ├── processors/
│   │   ├── cleaner.py             HTML, whitespace, dedup
│   │   ├── validator.py           Schema & quality checks
│   │   └── structurer.py          → KB entries + training rows
│   └── config.py                  All configurable settings
├── model_training/
│   ├── intent_classifier.py       spaCy or BERT backend
│   ├── response_generator.py      Template + NLU retrieval
│   ├── data_augmentation.py       Synonym/swap/delete + paraphrase
│   └── training_pipeline.py       Orchestration
├── output/
│   ├── trained_models/            Saved model artefacts
│   ├── knowledge_base.json        Structured health content
│   ├── training_data.csv          Labelled intent examples
│   └── embeddings/                Pre-computed vectors
├── scripts/
│   ├── run_collection.py          CLI: collect data
│   ├── train_models.py            CLI: train models
│   └── export_for_app.py          CLI: package for app
└── docs/
    └── DATA_PIPELINE.md           ← you are here
```

---

## Step 1 – Data Collection

### Running the scraper

```bash
python scripts/run_collection.py
```

Optional arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `--sources` | Space-separated list of sources to scrape | `red_cross cdc mayo_clinic` |
| `--output-dir` | Output directory | `output/` |
| `--dry-run` | Scrape but do not write files | disabled |

### Adding a new data source

1. Create `data_collection/scrapers/my_source_scraper.py` extending `BaseScraper`.
2. Implement `_scrape_endpoint(url)` to return a list of record dicts.
3. Add the source config to `DATA_SOURCES` in `data_collection/config.py`.
4. Import and register the scraper in `data_collection/scrapers/__init__.py`.
5. Add the scraper class to `_SCRAPER_MAP` in `scripts/run_collection.py`.

### Record schema

Each scraper must produce dicts with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `source` | str | Human-readable source name |
| `url` | str | Source URL |
| `title` | str | Page / section title |
| `content` | str | Extracted text content |
| `topics` | list[str] | Relevant topic tags |
| `category` | str | One of `ALLOWED_CATEGORIES` in config |

---

## Step 2 – Processing

The processing chain runs automatically after scraping:

```
Raw records → DataCleaner → DataValidator → DataStructurer → KB + CSV
```

### DataCleaner

- Decodes HTML entities
- Strips residual HTML tags and URLs
- Normalises Unicode (NFKC)
- Collapses whitespace
- Removes content shorter than `min_text_length`
- Deduplicates by content fingerprint

### DataValidator

- Checks all required fields are present
- Validates URL format
- Enforces content length bounds
- Verifies category is in the allowlist
- Rejects non-English content (> 30 % non-ASCII characters)

### DataStructurer

- Converts validated records to **knowledge-base entries** (with stable MD5 IDs)
- Produces **training rows** with heuristically inferred intent labels
- Intent inference uses keyword matching; the classifier later refines this

---

## Step 3 – Model Training

```bash
python scripts/train_models.py --backend spacy
```

Optional arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `--backend` | `spacy` or `bert` | `spacy` |
| `--training-data` | Path to CSV | `output/training_data.csv` |
| `--knowledge-base` | Path to JSON | `output/knowledge_base.json` |
| `--models-dir` | Output directory | `output/trained_models/` |

### spaCy backend (recommended for production)

Uses a **TF-IDF + Logistic Regression** pipeline from scikit-learn.  
Fast to train, small footprint, good accuracy for constrained domains.

```
pip install scikit-learn
```

### BERT backend (higher accuracy)

Fine-tunes `bert-base-uncased` using HuggingFace Transformers.  
Requires a GPU for practical training times.

```
pip install transformers torch
```

### Data augmentation

Before training, `DataAugmenter` expands the dataset using:

- **Synonym replacement** – swaps a random word with a WordNet synonym
- **Random deletion** – removes one non-leading word
- **Random swap** – transposes two adjacent words
- **Paraphrase templates** – regex-based surface-form variations

Control the expansion factor via `augmentation_factor` in `config.py`
(default: 3×).

---

## Step 4 – Export

```bash
python scripts/export_for_app.py --output-dir export/ [--embeddings]
```

Produces a self-contained `export/` directory containing:

- `intent_classifier/` – trained model artefacts
- `knowledge_base.json` – full KB
- `embeddings/` _(optional)_ – TF-IDF embeddings + index
- `manifest.json` – timestamps and SHA-256 checksums

The main pharmacy-connect application consumes this export directory.

---

## Configuration Reference (`data_collection/config.py`)

| Setting | Description |
|---------|-------------|
| `DATA_SOURCES` | Per-source URLs, topics, rate limits |
| `SCRAPER_SETTINGS` | User-agent, retries, page limits |
| `PROCESSOR_SETTINGS` | Length bounds, category allowlist |
| `MODEL_SETTINGS` | Hyper-parameters for all model components |
| `INTENT_LABELS` | Canonical set of intent labels |

---

## Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Optional (for BERT backend):

```bash
pip install transformers torch
```

Optional (for synonym augmentation):

```bash
pip install nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Extending the Pipeline

### Adding a new intent

1. Add the label to `INTENT_LABELS` in `config.py`.
2. Add training examples to `output/training_data.csv`.
3. Add response templates to `_TEMPLATES` in `model_training/response_generator.py`.
4. Add keyword hints to `_INTENT_KEYWORDS` in `data_collection/processors/structurer.py`.
5. Re-run training: `python scripts/train_models.py`.
