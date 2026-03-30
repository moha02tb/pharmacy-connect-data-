# Quick Start Guide

Get the PharmacyConnect data pipeline running in **~5 minutes**.

---

## Prerequisites

- Python 3.9+
- ~2 GB free disk space (for the MedDialog dataset cache)
- Internet access (first run only – dataset is cached locally)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/moha02tb/pharmacy-connect-data-
cd pharmacy-connect-data-

# Install dependencies
pip install -r requirements.txt
```

---

## Step 1 – Collect Training Data

```bash
python scripts/run_collection.py --max-records 65000
```

This will:
1. Download **OpenMed/MedDialog** from Hugging Face (~first run only)
2. Filter 65 000 pharmacy-relevant Q&A pairs
3. Map each pair to one of 13 intent labels
4. Write `output/knowledge_base.json` and `output/training_data.csv`

**Expected output:**

```
Collection summary:
  qa_pairs_loaded: 65000
  training_rows: 62340
  knowledge_base_entries: 65000
  drugbank_entries: 0
```

### Optional: Add DrugBank enrichment

1. Download DrugBank Open Data from https://go.drugbank.com/releases/latest
2. Run with the `--drugbank-file` flag:

```bash
python scripts/run_collection.py --drugbank-file path/to/drugbank_open.json
```

---

## Step 2 – Train the Intent Classifier

```bash
python scripts/train_models.py --backend spacy
```

The model is saved to `output/trained_models/intent_classifier_spacy/`.

---

## Step 3 – Export for the App

```bash
python scripts/export_for_app.py --output-dir export/
```

The `export/` directory contains everything the PharmacyConnect app needs.

---

## Quick Verification

```bash
# Run the unit tests
python -m unittest tests.test_meddialog_loader -v
```

All 32 tests should pass.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_DATASETS_CACHE` | Local cache directory for Hugging Face datasets | `~/.cache/huggingface/datasets` |
| `DRUGBANK_FILE` | Path to a local DrugBank JSON/CSV file | _(none)_ |

---

## Explore the Data (Notebook)

Open the interactive exploration notebook:

```bash
jupyter notebook notebooks/meddialog_exploration.ipynb
```

---

## Troubleshooting

### `ImportError: No module named 'datasets'`

```bash
pip install datasets>=4.0.0 huggingface-hub>=0.16.0
```

### Slow first run

The dataset is ~500 MB and is downloaded once then cached.  Subsequent runs
use the local cache and complete in seconds.

### `ConnectionError` / firewall issues

Set `HF_DATASETS_CACHE` to a directory containing a pre-downloaded copy of
the dataset, or download it on a machine with internet access and transfer it.

---

See **[docs/DATA_PIPELINE.md](DATA_PIPELINE.md)** for the full technical reference.
