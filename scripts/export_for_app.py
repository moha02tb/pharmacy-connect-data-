"""
export_for_app.py – Package trained models and knowledge base for the main
pharmacy-connect application.

Usage
-----
::

    python scripts/export_for_app.py [--output-dir export/]
                                     [--backend spacy|bert]

What this script produces
--------------------------
export/
├── intent_classifier/       ← copied from output/trained_models/
│   ├── model.pkl
│   └── pipeline_metadata.json
├── knowledge_base.json      ← full KB
├── embeddings/
│   ├── knowledge_base_embeddings.npy   ← sentence embeddings (optional)
│   └── embedding_index.json            ← row → KB id mapping
└── manifest.json            ← version info and file checksums
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Dict, List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_collection.config import (
    EMBEDDINGS_DIR,
    KNOWLEDGE_BASE_PATH,
    TRAINED_MODELS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_EXPORT_DIR = os.path.join(_PROJECT_ROOT, "export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pharmacy Connect – export for app")
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_EXPORT_DIR,
        help="Destination directory for exported artefacts (default: export/)",
    )
    parser.add_argument(
        "--backend",
        choices=["spacy", "bert"],
        default="spacy",
        help="Which intent-classifier backend to export (default: spacy)",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Generate and export knowledge-base sentence embeddings",
    )
    return parser.parse_args()


def export(output_dir: str, backend: str = "spacy", build_embeddings: bool = False) -> Dict:
    """
    Package all artefacts into *output_dir*.

    Returns a manifest dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest: Dict = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "files": {},
    }

    # ── 1. Copy intent classifier ─────────────────────────────────────────────
    src_model_dir = os.path.join(TRAINED_MODELS_DIR, f"intent_classifier_{backend}")
    dst_model_dir = os.path.join(output_dir, "intent_classifier")

    if os.path.isdir(src_model_dir):
        if os.path.exists(dst_model_dir):
            shutil.rmtree(dst_model_dir)
        shutil.copytree(src_model_dir, dst_model_dir)
        logger.info("Intent classifier copied to %s", dst_model_dir)
        for fname in os.listdir(dst_model_dir):
            fpath = os.path.join(dst_model_dir, fname)
            manifest["files"][f"intent_classifier/{fname}"] = _sha256(fpath)
    else:
        logger.warning(
            "Trained model not found at %s – run train_models.py first.", src_model_dir
        )

    # ── 2. Copy knowledge base ────────────────────────────────────────────────
    dst_kb = os.path.join(output_dir, "knowledge_base.json")
    if os.path.isfile(KNOWLEDGE_BASE_PATH):
        shutil.copy2(KNOWLEDGE_BASE_PATH, dst_kb)
        manifest["files"]["knowledge_base.json"] = _sha256(dst_kb)
        logger.info("Knowledge base copied to %s", dst_kb)
    else:
        logger.warning("Knowledge base not found at %s", KNOWLEDGE_BASE_PATH)

    # ── 3. Generate embeddings (optional) ─────────────────────────────────────
    if build_embeddings:
        emb_dir = os.path.join(output_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        _build_embeddings(dst_kb, emb_dir, manifest)

    # ── 4. Write manifest ─────────────────────────────────────────────────────
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    return manifest


def _build_embeddings(kb_path: str, emb_dir: str, manifest: Dict) -> None:
    """Generate sentence embeddings for all KB entries using scikit-learn TF-IDF."""
    if not os.path.isfile(kb_path):
        logger.warning("KB not found – skipping embedding generation.")
        return

    with open(kb_path, encoding="utf-8") as f:
        entries: List[Dict] = json.load(f)

    texts = [e.get("content", "") for e in entries]
    ids = [e.get("id", str(i)) for i, e in enumerate(entries)]

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=5000)
        matrix = vectorizer.fit_transform(texts).toarray()

        emb_path = os.path.join(emb_dir, "knowledge_base_embeddings.npy")
        np.save(emb_path, matrix)
        logger.info("Embeddings saved to %s (%s)", emb_path, matrix.shape)

        index_path = os.path.join(emb_dir, "embedding_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "shape": list(matrix.shape)}, f, indent=2)

        manifest["files"]["embeddings/knowledge_base_embeddings.npy"] = _sha256(emb_path)
        manifest["files"]["embeddings/embedding_index.json"] = _sha256(index_path)

    except ImportError:
        logger.warning("numpy/scikit-learn not installed – embeddings skipped.")


def _sha256(path: str) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    args = parse_args()
    result = export(
        output_dir=args.output_dir,
        backend=args.backend,
        build_embeddings=args.embeddings,
    )
    print("\n=== Export complete ===")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Files exported: {len(result['files'])}")
    print(f"  Exported at: {result['exported_at']}")
