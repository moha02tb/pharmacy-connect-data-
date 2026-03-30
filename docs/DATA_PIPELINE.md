# Data Pipeline – Technical Reference

This document describes the end-to-end data pipeline for the
**pharmacy-connect-data** repository, covering data collection, processing,
model training, and artefact export.

> **v2 Note** – Legacy web scrapers (Red Cross, CDC, Mayo Clinic) have been
> removed because their endpoints were unreliable (403 Forbidden / 404 / rate-
> limiting, yielding only 43 rows).  They are replaced by the
> **OpenMed/MedDialog** Hugging Face dataset (1.47 M doctor-patient
> conversations) and optional **DrugBank Open Data** enrichment.

---

## Overview

```
Data Sources                  Processing                   Model Training        Export
────────────────────────────────────────────────────────────────────────────────────────
OpenMed/MedDialog ──┐         ┌──────────────────────┐    ┌──────────────┐      ┌──────────┐
(1.47M Q&A pairs)   ├──────► │ IntentMapper           │   │ IntentCls    │──────► manifest  │
DrugBank (optional) ─┘         │ MedDialogProcessor    │   │ ResponseGen  │      │ KB copy  │
                               │ MedDialogKBBuilder    │   │ DataAugment  │      │ Embeds   │
                               └──────────────────────┘    └──────────────┘      └──────────┘
```

---

## Directory Structure

```
pharmacy-connect-data/
├── data_collection/
│   ├── scrapers/
│   │   ├── base_scraper.py          Abstract base (template for future scrapers)
│   │   ├── meddialog_loader.py      Load OpenMed/MedDialog from Hugging Face
│   │   ├── meddialog_kb_builder.py  Build KB entries from Q&A pairs
│   │   └── drugbank_loader.py       Load DrugBank drug metadata (optional)
│   ├── processors/
│   │   ├── cleaner.py               HTML, whitespace, dedup
│   │   ├── validator.py             Schema & quality checks
│   │   ├── structurer.py            → KB entries + training rows
│   │   ├── intent_mapper.py         Map Q&A to 13 pharmacy intents
│   │   └── meddialog_processor.py   Convert Q&A pairs to training rows
│   ├── config.py                    All configurable settings
│   └── meddialog_config.yaml        Dataset-specific configuration
├── model_training/
│   ├── intent_classifier.py         spaCy or BERT backend
│   ├── response_generator.py        Template + NLU retrieval
│   ├── data_augmentation.py         Synonym/swap/delete + paraphrase
│   └── training_pipeline.py         Orchestration
├── utils/
│   ├── logger.py                    Consistent logging
│   └── data_utils.py                Data manipulation helpers
├── output/
│   ├── trained_models/              Saved model artefacts
│   ├── knowledge_base.json          Structured health content
│   ├── training_data.csv            Labelled intent examples
│   └── embeddings/                  Pre-computed vectors
├── scripts/
│   ├── run_collection.py            CLI: collect data (MedDialog pipeline)
│   ├── train_models.py              CLI: train models
│   └── export_for_app.py            CLI: package for app
├── tests/
│   └── test_meddialog_loader.py     Unit tests for MedDialog integration
├── notebooks/
│   └── meddialog_exploration.ipynb  Interactive data exploration
└── docs/
    ├── DATA_PIPELINE.md             ← you are here
    └── QUICK_START.md               Quick start guide
```

---

## Step 1 – Data Collection

### Running the pipeline

```bash
python scripts/run_collection.py
```

Optional arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `--max-records` | Maximum Q&A pairs to extract | `65000` |
| `--output-dir` | Output directory | `output/` |
| `--drugbank-file` | Path to local DrugBank JSON/CSV | _(none)_ |
| `--dry-run` | Process but do not write files | disabled |
| `--no-filter` | Disable pharmacy-keyword filtering | disabled |

### Pipeline steps

```
Step 1: Load MedDialog (1.47M conversations → filtered Q&A pairs)
Step 2: Intent mapping + training row extraction + KB building
Step 3: Write knowledge_base.json + training_data.csv
```

### Data sources

#### OpenMed/MedDialog (primary)

- **Source**: Hugging Face – `OpenMed/MedDialog`
- **Size**: 1.47M real doctor-patient conversations
- **Access**: Automatic download via `datasets` library (cached locally)
- **Filtering**: Only pharmacy-relevant conversations are retained

#### DrugBank Open Data (optional enrichment)

- **Source**: https://go.drugbank.com/releases/latest (free tier, local download)
- **Size**: 13 000+ drug entries
- **Setup**: Download and set `--drugbank-file` or `DRUGBANK_FILE` env var

### Adding a new data source

1. Create `data_collection/scrapers/my_source_loader.py`.
2. Implement a `load()` method returning Q&A or KB record dicts.
3. Add the source config to `DATA_SOURCES` in `data_collection/config.py`.
4. Import and register the loader in `data_collection/scrapers/__init__.py`.
5. Add loading logic to `scripts/run_collection.py`.

### Training record schema

| Field | Type | Description |
|-------|------|-------------|
| `text` | str | Patient question (≤ 500 chars) |
| `intent` | str | One of the 13 intent labels |
| `source` | str | Dataset identifier |
| `category` | str | Topic category |

### Knowledge-base entry schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | 12-char MD5 fingerprint |
| `title` | str | Question or drug name |
| `content` | str | Doctor answer or drug info |
| `source` | str | Dataset or database name |
| `url` | str | Source URL (empty for dataset entries) |
| `category` | str | Topic category |
| `topics` | list[str] | Matched pharmacy keywords |

---

## Step 2 – Intent Mapping

The 13 pharmacy intent labels are:

| Intent | Description |
|--------|-------------|
| `medication_inquiry` | Questions about specific drugs |
| `drug_interaction_check` | Drug combination safety |
| `side_effects_inquiry` | Adverse reactions |
| `dosage_inquiry` | How much / how often to take |
| `pharmacy_location` | Finding a pharmacy |
| `prescription_refill` | Refilling prescriptions |
| `vaccination_info` | Vaccine information |
| `first_aid_guidance` | First aid instructions |
| `emergency_assistance` | Emergency situations |
| `general_health_question` | General health questions |
| `greeting` | Greetings |
| `farewell` | Farewells / thanks |
| `out_of_scope` | Out-of-scope queries |

Intent mapping uses keyword-matching with a configurable confidence threshold
(default: 0.3).  Records below the threshold receive `general_health_question`.

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

### BERT backend (higher accuracy)

Fine-tunes `bert-base-uncased` using HuggingFace Transformers.
Requires a GPU for practical training times.

```bash
pip install transformers torch
```

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

---

## Configuration Reference

### `data_collection/config.py`

| Setting | Description |
|---------|-------------|
| `DATA_SOURCES` | Per-source configuration (type, enabled flag, parameters) |
| `MEDDIALOG_SETTINGS` | Cache dir, confidence threshold, text length bounds |
| `PROCESSOR_SETTINGS` | Length bounds, category allowlist |
| `MODEL_SETTINGS` | Hyper-parameters for all model components |
| `INTENT_LABELS` | Canonical set of 13 intent labels |

### `data_collection/meddialog_config.yaml`

Fine-grained YAML configuration for the MedDialog loader, intent mapper, and
KB builder.  Values here mirror the Python config and can be used to override
defaults without editing Python source.

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

## Expected Results

| Metric | Before (scrapers) | After (MedDialog) |
|--------|-------------------|-------------------|
| Training Data | 53 rows | 65 000+ rows |
| Data Source | 3 broken scrapers | 1 reliable HF dataset |
| Setup Time | 30+ min (debugging) | ~5 min (cached) |
| Intent Labels | Manual | Auto-extracted |
| Production Ready | ❌ | ✅ |

---

## Extending the Pipeline

### Adding a new intent

1. Add the label to `INTENT_LABELS` in `config.py`.
2. Add keyword patterns to `_INTENT_KEYWORDS` in `processors/intent_mapper.py`.
3. Add response templates to `_TEMPLATES` in `model_training/response_generator.py`.
4. Re-run the pipeline and training: `python scripts/run_collection.py && python scripts/train_models.py`.
