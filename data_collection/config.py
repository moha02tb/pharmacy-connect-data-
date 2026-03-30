"""
Configuration for data collection sources, scraping settings, and output paths.
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
    "red_cross": {
        "name": "American Red Cross",
        "base_url": "https://www.redcross.org",
        "endpoints": [
            "/take-a-class/first-aid",
            "/get-help/how-to-prepare-for-emergencies/types-of-emergencies/medical",
        ],
        "topics": [
            "first aid",
            "emergency preparedness",
            "medication safety",
            "bleeding control",
            "CPR",
        ],
        "rate_limit_seconds": 2.0,
        "timeout_seconds": 30,
        "enabled": True,
    },
    "cdc": {
        "name": "Centers for Disease Control and Prevention",
        "base_url": "https://www.cdc.gov",
        "endpoints": [
            "/medication/index.html",
            "/niosh/topics/medication/default.html",
            "/drugoverdose/index.html",
            "/coronavirus/2019-ncov/vaccines/index.html",
        ],
        "topics": [
            "medication safety",
            "drug interactions",
            "vaccination",
            "prescription guidelines",
            "overdose prevention",
        ],
        "rate_limit_seconds": 2.0,
        "timeout_seconds": 30,
        "enabled": True,
    },
    "mayo_clinic": {
        "name": "Mayo Clinic",
        "base_url": "https://www.mayoclinic.org",
        "endpoints": [
            "/patient-care-and-health-information/consumer-health",
            "/diseases-conditions",
            "/drugs-supplements",
        ],
        "topics": [
            "drug information",
            "side effects",
            "dosage",
            "drug interactions",
            "health conditions",
            "supplements",
        ],
        "rate_limit_seconds": 3.0,
        "timeout_seconds": 30,
        "enabled": True,
    },
}

# ── Scraper settings ──────────────────────────────────────────────────────────
SCRAPER_SETTINGS = {
    "user_agent": (
        "Mozilla/5.0 (compatible; PharmacyConnectBot/1.0; "
        "+https://github.com/moha02tb/pharmacy-connect-data-)"
    ),
    "max_retries": 3,
    "backoff_factor": 2.0,
    "max_pages_per_source": 50,
    "min_content_length": 100,
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
