# Embeddings

Pre-computed sentence / document embeddings are stored here by the export
script (`scripts/export_for_app.py`).

Files:
- `knowledge_base_embeddings.npy` – NumPy array of KB entry embeddings
- `embedding_index.json`          – Mapping from row index → KB entry id

This directory is tracked in git (via this README) but binary `.npy` files
are excluded via `.gitignore`.
