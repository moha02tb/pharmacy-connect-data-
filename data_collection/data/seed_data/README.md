# Seed Data

This directory holds optional seed / bootstrap data files for the
Pharmacy Connect data-collection pipeline.

## Supported formats

| File pattern | Description |
|---|---|
| `*.json` | JSON array of Q&A objects (keys: `question`, `answer`, `category`) |
| `*.csv`  | CSV with columns `question`, `answer`, `category` |

## Usage

Place any local seed data files here before running the pipeline.  The
pipeline will merge seed records with data fetched from Hugging Face so
that the final `knowledge_base.json` and `training_data.csv` include both
sources.

If no seed files are present the pipeline continues normally using only
the MedDialog dataset (and the optional DrugBank file).
