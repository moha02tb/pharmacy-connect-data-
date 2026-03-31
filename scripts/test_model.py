"""
test_model.py – Test the trained pharmacy-connect intent classifier.

Usage
-----
::

    # Run predefined test queries and show predictions + confidence
    python scripts/test_model.py

    # Specify a different backend or model directory
    python scripts/test_model.py --backend spacy --models-dir output/trained_models/

    # Launch an interactive REPL to type your own queries
    python scripts/test_model.py --interactive

"""

import argparse
import logging
import os
import sys
from typing import Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_collection.config import (
    KNOWLEDGE_BASE_PATH,
    TRAINED_MODELS_DIR,
)
from model_training.intent_classifier import IntentClassifier
from model_training.response_generator import ResponseGenerator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Sample queries – one or more per intent label ─────────────────────────────

_TEST_QUERIES: Dict[str, list] = {
    "greeting": [
        "Hello, can you help me?",
        "Hi there!",
    ],
    "farewell": [
        "Goodbye, thanks for your help.",
        "See you later!",
    ],
    "medication_inquiry": [
        "What is ibuprofen used for?",
        "Tell me about metformin.",
        "Can you explain what amoxicillin does?",
    ],
    "drug_interaction_check": [
        "Can I take aspirin and warfarin together?",
        "Are there any interactions between lisinopril and ibuprofen?",
    ],
    "side_effects_inquiry": [
        "What are the side effects of metoprolol?",
        "Does prednisone cause weight gain?",
    ],
    "dosage_inquiry": [
        "What is the correct dose of paracetamol for adults?",
        "How many mg of ibuprofen can I take per day?",
    ],
    "pharmacy_location": [
        "Where is the nearest pharmacy?",
        "Find a pharmacy close to me.",
    ],
    "prescription_refill": [
        "How do I refill my prescription?",
        "I need to renew my medication prescription.",
    ],
    "vaccination_info": [
        "What vaccines do I need for flu season?",
        "Is the COVID-19 booster available at pharmacies?",
    ],
    "first_aid_guidance": [
        "How do I treat a minor burn?",
        "What should I do if someone is choking?",
    ],
    "emergency_assistance": [
        "I think I accidentally took too many pills.",
        "Someone is having a severe allergic reaction.",
    ],
    "general_health_question": [
        "How can I lower my blood pressure naturally?",
        "What are the symptoms of diabetes?",
    ],
    "out_of_scope": [
        "What is the weather like today?",
        "Who won the football match last night?",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pharmacy Connect – test the trained intent classifier"
    )
    parser.add_argument(
        "--backend",
        choices=["spacy", "bert"],
        default="spacy",
        help="Backend used during training (default: spacy)",
    )
    parser.add_argument(
        "--models-dir",
        default=TRAINED_MODELS_DIR,
        help="Directory containing trained models (default: output/trained_models/)",
    )
    parser.add_argument(
        "--knowledge-base",
        default=KNOWLEDGE_BASE_PATH,
        help="Path to knowledge_base.json (default: output/knowledge_base.json)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive REPL to test the model with custom queries",
    )
    parser.add_argument(
        "--no-response",
        action="store_true",
        help="Skip response generation and only show intent predictions",
    )
    return parser.parse_args()


def load_classifier(models_dir: str, backend: str) -> IntentClassifier:
    """Load the saved IntentClassifier from *models_dir*."""
    model_path = os.path.join(models_dir, f"intent_classifier_{backend}")
    if not os.path.isdir(model_path):
        print(
            f"\n[ERROR] Model directory not found: {model_path}\n"
            "        Run 'python scripts/train_models.py' first to train the model."
        )
        sys.exit(1)

    classifier = IntentClassifier(backend=backend)
    classifier.load(model_path)
    return classifier


def load_response_generator(knowledge_base_path: str) -> ResponseGenerator:
    """Load the ResponseGenerator (with KB if available)."""
    return ResponseGenerator(knowledge_base_path=knowledge_base_path)


def predict_and_show(
    query: str,
    classifier: IntentClassifier,
    generator: ResponseGenerator,
    show_response: bool = True,
) -> str:
    """Print the prediction details for *query* and return the predicted intent."""
    predicted = classifier.predict(query)
    scores = classifier.predict_proba(query)

    # Sort by confidence descending
    top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = " | ".join(f"{label}: {conf:.2%}" for label, conf in top_scores)

    print(f"  Query    : {query}")
    print(f"  Predicted: {predicted}")
    print(f"  Top-3    : {top_str}")

    if show_response:
        response = generator.generate(query=query, intent=predicted)
        print(f"  Response : {response}")

    print()
    return predicted


def run_predefined_tests(
    classifier: IntentClassifier,
    generator: ResponseGenerator,
    show_response: bool,
) -> None:
    """Run all predefined test queries and print a summary."""
    print("\n" + "=" * 70)
    print("  PREDEFINED TEST QUERIES")
    print("=" * 70)

    correct = 0
    total = 0
    mismatches = []

    for expected_intent, queries in _TEST_QUERIES.items():
        print(f"\n── Intent: {expected_intent} ──")
        for query in queries:
            predicted = predict_and_show(query, classifier, generator, show_response)
            total += 1
            if predicted == expected_intent:
                correct += 1
            else:
                mismatches.append((query, expected_intent, predicted))

    print("=" * 70)
    accuracy = correct / total if total else 0.0
    print(f"  Test accuracy on sample queries: {correct}/{total} ({accuracy:.1%})")

    if mismatches:
        print("\n  Misclassified queries:")
        for query, expected, predicted in mismatches:
            print(f"    [{expected}] → '{predicted}':  \"{query}\"")

    print("=" * 70 + "\n")


def run_interactive(
    classifier: IntentClassifier,
    generator: ResponseGenerator,
    show_response: bool,
) -> None:
    """Start an interactive REPL for manual testing."""
    print("\n" + "=" * 70)
    print("  INTERACTIVE MODE  (type 'quit' or 'exit' to stop)")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        predict_and_show(query, classifier, generator, show_response)


def main() -> None:
    args = parse_args()

    print(f"Loading model (backend={args.backend}) from {args.models_dir} …")
    classifier = load_classifier(args.models_dir, args.backend)

    kb_status = (
        "loaded"
        if os.path.isfile(args.knowledge_base)
        else "not found – running in template-only mode"
    )
    print(f"Loading response generator … knowledge base {kb_status}")
    generator = load_response_generator(args.knowledge_base)

    show_response = not args.no_response

    if args.interactive:
        run_interactive(classifier, generator, show_response)
    else:
        run_predefined_tests(classifier, generator, show_response)


if __name__ == "__main__":
    main()
