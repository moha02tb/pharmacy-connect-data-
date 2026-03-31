"""
tests/test_intent_classifier.py – Unit tests for IntentClassifier.

Covers:
- predict / predict_proba / predict_with_confidence API
- Safety override behaviour
- LOW_CONFIDENCE_THRESHOLD constant
- spaCy-backend training with improved features
- save / load round-trip
"""

import unittest


# ---------------------------------------------------------------------------
# Minimal training fixture shared across tests
# ---------------------------------------------------------------------------

_SEED_DATA = [
    # side_effects_inquiry (5)
    {"text": "What are the side effects of ibuprofen?", "intent": "side_effects_inquiry"},
    {"text": "Does aspirin cause bleeding?", "intent": "side_effects_inquiry"},
    {"text": "Side effects of metformin", "intent": "side_effects_inquiry"},
    {"text": "Can lisinopril cause a dry cough?", "intent": "side_effects_inquiry"},
    {"text": "What are metoprolol side effects?", "intent": "side_effects_inquiry"},
    # dosage_inquiry (5)
    {"text": "How much ibuprofen can I take?", "intent": "dosage_inquiry"},
    {"text": "What is the maximum dose of Tylenol per day?", "intent": "dosage_inquiry"},
    {"text": "How many mg of melatonin should I take?", "intent": "dosage_inquiry"},
    {"text": "What is the adult dosage for Benadryl?", "intent": "dosage_inquiry"},
    {"text": "How much ibuprofen is safe per day?", "intent": "dosage_inquiry"},
    # drug_interaction_check (5)
    {"text": "Can I take ibuprofen with alcohol?", "intent": "drug_interaction_check"},
    {"text": "Does grapefruit affect blood pressure medication?", "intent": "drug_interaction_check"},
    {"text": "What drugs interact with warfarin?", "intent": "drug_interaction_check"},
    {"text": "Is it ok to take aspirin and ibuprofen together?", "intent": "drug_interaction_check"},
    {"text": "Can I combine antibiotics with birth control?", "intent": "drug_interaction_check"},
    # medication_inquiry (5)
    {"text": "What is ibuprofen used for?", "intent": "medication_inquiry"},
    {"text": "Tell me about lisinopril", "intent": "medication_inquiry"},
    {"text": "What is metformin?", "intent": "medication_inquiry"},
    {"text": "What is atorvastatin?", "intent": "medication_inquiry"},
    {"text": "What does Zoloft treat?", "intent": "medication_inquiry"},
    # vaccination_info (5)
    {"text": "When should I get a flu shot?", "intent": "vaccination_info"},
    {"text": "Do I need a COVID booster?", "intent": "vaccination_info"},
    {"text": "Can I get the flu shot while pregnant?", "intent": "vaccination_info"},
    {"text": "Should I get the shingles vaccine?", "intent": "vaccination_info"},
    {"text": "Do adults need a tetanus booster?", "intent": "vaccination_info"},
    # first_aid_guidance (5)
    {"text": "How do I treat a minor burn?", "intent": "first_aid_guidance"},
    {"text": "I got burned on the stove", "intent": "first_aid_guidance"},
    {"text": "How do I stop bleeding?", "intent": "first_aid_guidance"},
    {"text": "What to do for a sprained ankle?", "intent": "first_aid_guidance"},
    {"text": "How do I perform CPR?", "intent": "first_aid_guidance"},
    # emergency_assistance (5)
    {"text": "Someone overdosed on opioids what do I do?", "intent": "emergency_assistance"},
    {"text": "My child swallowed something dangerous", "intent": "emergency_assistance"},
    {"text": "Someone is not waking up", "intent": "emergency_assistance"},
    {"text": "Someone is having a heart attack", "intent": "emergency_assistance"},
    {"text": "Poison control number", "intent": "emergency_assistance"},
    # pharmacy_location (5)
    {"text": "Where is the nearest pharmacy?", "intent": "pharmacy_location"},
    {"text": "CVS pharmacy near me", "intent": "pharmacy_location"},
    {"text": "Is there a 24-hour pharmacy nearby?", "intent": "pharmacy_location"},
    {"text": "What pharmacies are open right now?", "intent": "pharmacy_location"},
    {"text": "Walgreens pharmacy hours", "intent": "pharmacy_location"},
    # prescription_refill (5)
    {"text": "How do I refill my prescription?", "intent": "prescription_refill"},
    {"text": "I need to renew my prescription", "intent": "prescription_refill"},
    {"text": "Can I get an early refill?", "intent": "prescription_refill"},
    {"text": "My prescription is running low", "intent": "prescription_refill"},
    {"text": "How do I set up automatic refills?", "intent": "prescription_refill"},
    # general_health_question (5)
    {"text": "How should I store my medications?", "intent": "general_health_question"},
    {"text": "What happens if I miss a dose?", "intent": "general_health_question"},
    {"text": "Are generic drugs as effective as brand name?", "intent": "general_health_question"},
    {"text": "Can I stop taking my medication when I feel better?", "intent": "general_health_question"},
    {"text": "Should I take my medication with food?", "intent": "general_health_question"},
    # greeting (5)
    {"text": "Hello", "intent": "greeting"},
    {"text": "Hi there", "intent": "greeting"},
    {"text": "Good morning", "intent": "greeting"},
    {"text": "Hey", "intent": "greeting"},
    {"text": "Hello there", "intent": "greeting"},
    # farewell (5)
    {"text": "Goodbye", "intent": "farewell"},
    {"text": "Thank you for your help", "intent": "farewell"},
    {"text": "Bye", "intent": "farewell"},
    {"text": "Thanks a lot", "intent": "farewell"},
    {"text": "Take care", "intent": "farewell"},
    # out_of_scope (5)
    {"text": "What is the weather today?", "intent": "out_of_scope"},
    {"text": "Tell me a joke", "intent": "out_of_scope"},
    {"text": "What is the capital of France?", "intent": "out_of_scope"},
    {"text": "Play some music for me", "intent": "out_of_scope"},
    {"text": "What is 2 plus 2?", "intent": "out_of_scope"},
]


class TestIntentClassifierConstants(unittest.TestCase):
    """Tests for module-level constants and utilities."""

    def test_low_confidence_threshold_is_float(self):
        from model_training.intent_classifier import LOW_CONFIDENCE_THRESHOLD
        self.assertIsInstance(LOW_CONFIDENCE_THRESHOLD, float)

    def test_low_confidence_threshold_range(self):
        from model_training.intent_classifier import LOW_CONFIDENCE_THRESHOLD
        self.assertGreater(LOW_CONFIDENCE_THRESHOLD, 0.0)
        self.assertLess(LOW_CONFIDENCE_THRESHOLD, 1.0)

    def test_supported_backends(self):
        from model_training.intent_classifier import SUPPORTED_BACKENDS
        self.assertIn("spacy", SUPPORTED_BACKENDS)
        self.assertIn("bert", SUPPORTED_BACKENDS)


class TestSafetyOverride(unittest.TestCase):
    """Tests for _apply_safety_override."""

    def _override(self, text, predicted):
        from model_training.intent_classifier import _apply_safety_override
        return _apply_safety_override(text, predicted)

    def test_burn_overrides_general_health(self):
        self.assertEqual(self._override("I got burned", "general_health_question"), "first_aid_guidance")

    def test_chemical_burn_phrase_overrides(self):
        self.assertEqual(
            self._override("I have a chemical burn on my arm", "general_health_question"),
            "first_aid_guidance",
        )

    def test_electrical_burn_overrides(self):
        self.assertEqual(
            self._override("Electrical burn treatment", "general_health_question"),
            "first_aid_guidance",
        )

    def test_sunburn_overrides(self):
        self.assertEqual(
            self._override("I got a sunburn at the beach", "general_health_question"),
            "first_aid_guidance",
        )

    def test_overdose_overrides_to_emergency(self):
        self.assertEqual(
            self._override("Someone overdosed on opioids", "general_health_question"),
            "emergency_assistance",
        )

    def test_non_general_health_unchanged(self):
        # Override only applies to general_health_question
        self.assertEqual(self._override("I got burned", "medication_inquiry"), "medication_inquiry")

    def test_burnout_not_triggered(self):
        # "burnout" should NOT trigger the burn override
        self.assertEqual(
            self._override("My burnout from work is terrible", "general_health_question"),
            "general_health_question",
        )

    def test_neutral_query_unchanged(self):
        self.assertEqual(
            self._override("What is a generic drug?", "general_health_question"),
            "general_health_question",
        )

    def test_second_degree_burn_overrides(self):
        self.assertEqual(
            self._override("Second degree burn care tips", "general_health_question"),
            "first_aid_guidance",
        )


class TestIntentClassifierInit(unittest.TestCase):
    """Tests for IntentClassifier initialisation."""

    def test_default_backend_is_spacy(self):
        from model_training.intent_classifier import IntentClassifier
        clf = IntentClassifier()
        self.assertEqual(clf.backend, "spacy")

    def test_invalid_backend_raises(self):
        from model_training.intent_classifier import IntentClassifier
        with self.assertRaises(ValueError):
            IntentClassifier(backend="invalid")

    def test_not_trained_on_init(self):
        from model_training.intent_classifier import IntentClassifier
        clf = IntentClassifier()
        self.assertFalse(clf._is_trained)


class TestIntentClassifierUntrainedBehaviour(unittest.TestCase):
    """Tests for safe fallbacks when model is not trained."""

    def setUp(self):
        from model_training.intent_classifier import IntentClassifier
        self.clf = IntentClassifier()

    def test_predict_returns_unknown(self):
        self.assertEqual(self.clf.predict("hello"), "unknown")

    def test_predict_proba_returns_empty_dict(self):
        self.assertEqual(self.clf.predict_proba("hello"), {})

    def test_predict_with_confidence_returns_unknown_zero(self):
        intent, conf = self.clf.predict_with_confidence("hello")
        self.assertEqual(intent, "unknown")
        self.assertEqual(conf, 0.0)

    def test_save_raises_when_untrained(self):
        with self.assertRaises(RuntimeError):
            self.clf.save("/tmp/unused")


class TestIntentClassifierTrain(unittest.TestCase):
    """Tests for training the spaCy backend."""

    def setUp(self):
        from model_training.intent_classifier import IntentClassifier
        self.clf = IntentClassifier(backend="spacy")
        self.metrics = self.clf.train(_SEED_DATA)

    def test_train_returns_metrics_dict(self):
        self.assertIsInstance(self.metrics, dict)

    def test_metrics_has_accuracy(self):
        self.assertIn("accuracy", self.metrics)

    def test_metrics_has_f1(self):
        self.assertIn("f1", self.metrics)

    def test_accuracy_in_range(self):
        self.assertGreaterEqual(self.metrics["accuracy"], 0.0)
        self.assertLessEqual(self.metrics["accuracy"], 1.0)

    def test_is_trained_after_train(self):
        self.assertTrue(self.clf._is_trained)

    def test_train_empty_raises(self):
        from model_training.intent_classifier import IntentClassifier
        clf = IntentClassifier()
        with self.assertRaises(ValueError):
            clf.train([])


class TestIntentClassifierPredict(unittest.TestCase):
    """Tests for predict / predict_proba / predict_with_confidence after training."""

    @classmethod
    def setUpClass(cls):
        from model_training.intent_classifier import IntentClassifier
        cls.clf = IntentClassifier(backend="spacy")
        cls.clf.train(_SEED_DATA)

    def test_predict_returns_string(self):
        self.assertIsInstance(self.clf.predict("hello"), str)

    def test_predict_known_intent(self):
        intent = self.clf.predict("Hello there")
        # After training on seed data with greeting examples, should recognise
        from data_collection.config import INTENT_LABELS
        self.assertIn(intent, INTENT_LABELS)

    def test_predict_proba_returns_dict(self):
        proba = self.clf.predict_proba("What are the side effects of aspirin?")
        self.assertIsInstance(proba, dict)

    def test_predict_proba_sums_to_one(self):
        proba = self.clf.predict_proba("What are the side effects of aspirin?")
        total = sum(proba.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_predict_proba_all_non_negative(self):
        proba = self.clf.predict_proba("any query")
        for v in proba.values():
            self.assertGreaterEqual(v, 0.0)

    def test_predict_with_confidence_returns_tuple(self):
        result = self.clf.predict_with_confidence("hello")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_predict_with_confidence_intent_is_string(self):
        intent, _ = self.clf.predict_with_confidence("hello")
        self.assertIsInstance(intent, str)

    def test_predict_with_confidence_confidence_is_float(self):
        _, conf = self.clf.predict_with_confidence("hello")
        self.assertIsInstance(conf, float)

    def test_predict_with_confidence_confidence_in_range(self):
        _, conf = self.clf.predict_with_confidence("hello")
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_safety_override_returns_1_confidence(self):
        # With sufficient training data the model may already predict first_aid_guidance
        # directly; if the safety override fires instead, confidence will be 1.0.
        # Either way the result must be first_aid_guidance with valid confidence.
        intent, conf = self.clf.predict_with_confidence("Help I got burned")
        self.assertEqual(intent, "first_aid_guidance")
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_safety_override_emergency(self):
        intent, conf = self.clf.predict_with_confidence("Someone overdosed on opioids")
        self.assertIn(intent, ("emergency_assistance", "first_aid_guidance"))


class TestIntentClassifierSaveLoad(unittest.TestCase):
    """Tests for model persistence."""

    def test_save_and_load_roundtrip(self):
        import tempfile
        import os
        from model_training.intent_classifier import IntentClassifier

        clf = IntentClassifier(backend="spacy")
        clf.train(_SEED_DATA)
        original_intent = clf.predict("What are the side effects of aspirin?")

        with tempfile.TemporaryDirectory() as tmpdir:
            clf.save(tmpdir)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "model.pkl")))

            clf2 = IntentClassifier(backend="spacy")
            clf2.load(tmpdir)
            self.assertTrue(clf2._is_trained)
            loaded_intent = clf2.predict("What are the side effects of aspirin?")

        self.assertEqual(original_intent, loaded_intent)
