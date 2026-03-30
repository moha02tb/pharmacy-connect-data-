"""
Configuration for data collection sources and output paths.

Legacy web scrapers (Red Cross, CDC, Mayo Clinic) have been replaced by the
OpenMed/MedDialog Hugging Face dataset and the optional DrugBank Open Data
file loader.
"""

import os

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TRAINED_MODELS_DIR = os.path.join(OUTPUT_DIR, "trained_models")
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, "embeddings")

# ── Output file paths ─────────────────────────────────────────────────────────
KNOWLEDGE_BASE_PATH = os.path.join(OUTPUT_DIR, "knowledge_base.json")
TRAINING_DATA_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")

# ── Data source configurations ────────────────────────────────────────────────
DATA_SOURCES = {
    "meddialog": {
        "name": "OpenMed/MedDialog",
        "type": "huggingface_dataset",
        "dataset_id": "OpenMed/MedDialog",
        "split": "train",
        "filter_pharmacy": True,
        "max_records": 65_000,
        "enabled": True,
    },
    "drugbank": {
        "name": "DrugBank Open Data",
        "type": "json_api",
        # Set this to a local file path after downloading from
        # https://go.drugbank.com/releases/latest
        "file_path": os.environ.get("DRUGBANK_FILE", ""),
        "max_records": 13_000,
        "enabled": bool(os.environ.get("DRUGBANK_FILE", "")),
    },
    # Legacy scrapers – disabled; endpoints were unreliable
    "red_cross": {"enabled": False},
    "cdc": {"enabled": False},
    "mayo_clinic": {"enabled": False},
}

# ── MedDialog loader settings ─────────────────────────────────────────────────
MEDDIALOG_SETTINGS = {
    # HF_DATASETS_CACHE: custom override for the Hugging Face datasets cache
    # directory. When unset, the default (~/.cache/huggingface/datasets) is used.
    "cache_dir": os.environ.get("HF_DATASETS_CACHE", None),
    "confidence_threshold": 0.3,
    "max_text_length": 500,
    "min_text_length": 10,
    "include_answer_in_text": False,
}

# ── Processor settings ────────────────────────────────────────────────────────
PROCESSOR_SETTINGS = {
    "min_text_length": 20,
    "max_text_length": 2000,
    "allowed_categories": [
        "medication",
        "pharmacy",
        "first_aid",
        "vaccination",
        "drug_interaction",
        "dosage",
        "side_effects",
        "emergency",
        "general_health",
        "pharmacy_qa",
    ],
    "language": "en",
}

# ── Model training settings ───────────────────────────────────────────────────
MODEL_SETTINGS = {
    "intent_classifier": {
        "model_name": "en_core_web_sm",
        "bert_model": "bert-base-uncased",
        "max_length": 128,
        "test_size": 0.2,
        "random_state": 42,
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 2e-5,
    },
    "response_generator": {
        "max_response_length": 500,
        "similarity_threshold": 0.05,
        "top_k_responses": 3,
    },
    "data_augmentation": {
        "augmentation_factor": 3,
        "synonym_replacement_prob": 0.2,
        "random_deletion_prob": 0.1,
        "random_swap_prob": 0.1,
    },
}

# ── Intent labels ─────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "medication_inquiry",
    "drug_interaction_check",
    "side_effects_inquiry",
    "dosage_inquiry",
    "pharmacy_location",
    "prescription_refill",
    "vaccination_info",
    "first_aid_guidance",
    "emergency_assistance",
    "general_health_question",
    "greeting",
    "farewell",
    "out_of_scope",
]
