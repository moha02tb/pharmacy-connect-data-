"""
tests/test_response_generator.py – Unit tests for ResponseGenerator.

Covers:
- All 14 intent templates are present (including 'low_confidence')
- generate() returns a non-empty string
- Confidence-based routing to low_confidence template
- Safety intents are never re-routed on low confidence
- fill_template handles missing {medication} placeholder gracefully
- KB retrieval with title + content
- add_template API
- similarity_threshold default raised to 0.10
"""

import unittest


class TestTemplatesPresent(unittest.TestCase):
    """All expected intent templates including low_confidence must exist."""

    def _templates(self):
        from model_training.response_generator import _TEMPLATES
        return _TEMPLATES

    def test_all_intents_have_templates(self):
        from data_collection.config import INTENT_LABELS
        templates = self._templates()
        for intent in INTENT_LABELS:
            self.assertIn(intent, templates, f"Missing template for intent: {intent}")

    def test_low_confidence_template_exists(self):
        self.assertIn("low_confidence", self._templates())

    def test_each_intent_has_at_least_two_templates(self):
        templates = self._templates()
        # safety-critical and simple intents may have 1; most should have 2+
        two_plus = [
            "medication_inquiry", "drug_interaction_check", "side_effects_inquiry",
            "dosage_inquiry", "vaccination_info", "first_aid_guidance",
            "emergency_assistance", "pharmacy_location", "prescription_refill",
            "out_of_scope", "general_health_question", "low_confidence",
            "greeting", "farewell",
        ]
        for intent in two_plus:
            self.assertGreaterEqual(
                len(templates.get(intent, [])), 2,
                f"Intent '{intent}' should have ≥ 2 templates",
            )

    def test_kb_snippet_placeholder_in_relevant_intents(self):
        """Intents that need KB retrieval must include the {kb_snippet} placeholder."""
        templates = self._templates()
        kb_intents = [
            "medication_inquiry", "drug_interaction_check", "side_effects_inquiry",
            "dosage_inquiry", "vaccination_info", "first_aid_guidance",
            "emergency_assistance", "general_health_question",
        ]
        for intent in kb_intents:
            first_template = templates[intent][0]
            self.assertIn(
                "{kb_snippet}", first_template,
                f"First template for '{intent}' should contain {{kb_snippet}}",
            )


class TestResponseGeneratorInit(unittest.TestCase):
    """Tests for ResponseGenerator initialisation."""

    def _make_gen(self, **kwargs):
        from model_training.response_generator import ResponseGenerator
        return ResponseGenerator(**kwargs)

    def test_default_similarity_threshold(self):
        gen = self._make_gen()
        # Default raised from 0.05 to 0.10 for fewer irrelevant results
        self.assertGreaterEqual(gen.similarity_threshold, 0.10)

    def test_settings_override_threshold(self):
        gen = self._make_gen(settings={"similarity_threshold": 0.25})
        self.assertEqual(gen.similarity_threshold, 0.25)

    def test_no_kb_on_init(self):
        gen = self._make_gen()
        self.assertEqual(len(gen._knowledge_base), 0)

    def test_missing_kb_file_does_not_raise(self):
        gen = self._make_gen(knowledge_base_path="/nonexistent/kb.json")
        self.assertEqual(len(gen._knowledge_base), 0)


class TestGenerateWithoutKB(unittest.TestCase):
    """Tests for generate() in template-only mode (no KB loaded)."""

    @classmethod
    def setUpClass(cls):
        from model_training.response_generator import ResponseGenerator
        cls.gen = ResponseGenerator()

    def test_generate_returns_string(self):
        self.assertIsInstance(self.gen.generate("hello", "greeting"), str)

    def test_generate_non_empty(self):
        self.assertGreater(len(self.gen.generate("hello", "greeting")), 0)

    def test_generate_greeting(self):
        resp = self.gen.generate("Hi", "greeting")
        self.assertIn("PharmacyConnect", resp)

    def test_generate_farewell(self):
        resp = self.gen.generate("Goodbye", "farewell")
        self.assertTrue(len(resp) > 0)

    def test_generate_emergency_contains_911(self):
        resp = self.gen.generate("Someone is overdosing", "emergency_assistance")
        self.assertIn("911", resp)

    def test_generate_emergency_contains_poison_control(self):
        resp = self.gen.generate("Someone overdosed", "emergency_assistance")
        self.assertIn("1-800-222-1222", resp)

    def test_generate_first_aid_contains_911(self):
        resp = self.gen.generate("I got burned", "first_aid_guidance")
        self.assertIn("911", resp)

    def test_generate_prescription_refill_contains_options(self):
        resp = self.gen.generate("I need to refill", "prescription_refill")
        self.assertIn("pharmacy", resp.lower())

    def test_generate_out_of_scope(self):
        resp = self.gen.generate("Tell me a joke", "out_of_scope")
        self.assertIn("pharmacy", resp.lower())

    def test_generate_unknown_intent_returns_fallback(self):
        resp = self.gen.generate("some query", "nonexistent_intent")
        self.assertGreater(len(resp), 0)

    def test_generate_truncates_long_response(self):
        gen = self.gen.__class__(settings={"max_response_length": 50})
        resp = gen.generate("detailed question", "prescription_refill")
        self.assertLessEqual(len(resp), 55)  # small slack for ellipsis


class TestConfidenceRouting(unittest.TestCase):
    """Tests for confidence-based routing to low_confidence template."""

    @classmethod
    def setUpClass(cls):
        from model_training.response_generator import ResponseGenerator
        cls.gen = ResponseGenerator()

    def test_low_confidence_routes_to_clarification(self):
        resp = self.gen.generate("query", "medication_inquiry", confidence=0.20)
        # Should return low_confidence response asking for clarification
        self.assertTrue(
            "clarif" in resp.lower() or "understand" in resp.lower() or "rephrase" in resp.lower(),
            f"Expected clarification response, got: {resp}",
        )

    def test_high_confidence_uses_normal_template(self):
        resp = self.gen.generate("hello", "greeting", confidence=0.95)
        self.assertIn("PharmacyConnect", resp)

    def test_confidence_at_threshold_uses_normal_template(self):
        from model_training.response_generator import _LOW_CONFIDENCE_THRESHOLD
        resp = self.gen.generate("hello", "greeting", confidence=_LOW_CONFIDENCE_THRESHOLD)
        self.assertIn("PharmacyConnect", resp)

    def test_confidence_just_below_threshold_routes_to_clarification(self):
        from model_training.response_generator import _LOW_CONFIDENCE_THRESHOLD
        resp = self.gen.generate("query", "general_health_question", confidence=_LOW_CONFIDENCE_THRESHOLD - 0.01)
        self.assertTrue(
            "clarif" in resp.lower() or "understand" in resp.lower() or "rephrase" in resp.lower(),
        )

    def test_emergency_never_rerouted_on_low_confidence(self):
        resp = self.gen.generate("someone overdosed", "emergency_assistance", confidence=0.10)
        self.assertIn("911", resp)

    def test_first_aid_never_rerouted_on_low_confidence(self):
        resp = self.gen.generate("I got burned", "first_aid_guidance", confidence=0.05)
        self.assertIn("911", resp)

    def test_no_confidence_uses_normal_template(self):
        resp = self.gen.generate("hello", "greeting")
        self.assertIn("PharmacyConnect", resp)


class TestFillTemplate(unittest.TestCase):
    """Tests for _fill_template static method."""

    def _fill(self, template, entities, kb_snippet):
        from model_training.response_generator import ResponseGenerator
        return ResponseGenerator._fill_template(template, entities, kb_snippet)

    def test_fills_kb_snippet(self):
        result = self._fill("Here: {kb_snippet}", {}, "ibuprofen info")
        self.assertEqual(result, "Here: ibuprofen info")

    def test_fills_entity(self):
        result = self._fill("Info about {medication}", {"medication": "aspirin"}, "")
        self.assertEqual(result, "Info about aspirin")

    def test_missing_entity_placeholder_removed_cleanly(self):
        # "about {medication}" should not leave dangling "about ," in output
        result = self._fill(
            "For information about {medication}, I recommend consulting your pharmacist.",
            {},
            "",
        )
        self.assertNotIn("{medication}", result)
        self.assertNotIn("about ,", result)
        self.assertNotIn("about  ,", result)

    def test_empty_kb_snippet_replaced_with_fallback(self):
        result = self._fill("Content: {kb_snippet}", {}, "")
        self.assertIn("No relevant information found", result)

    def test_none_entities_does_not_raise(self):
        result = self._fill("Hello {name}", {}, "")
        self.assertNotIn("{name}", result)

    def test_multiple_entities_filled(self):
        result = self._fill("{drug} at {dosage}", {"drug": "aspirin", "dosage": "100mg"}, "")
        self.assertEqual(result, "aspirin at 100mg")


class TestAddTemplate(unittest.TestCase):
    """Tests for the add_template API."""

    def test_add_new_intent_template(self):
        from model_training.response_generator import ResponseGenerator, _TEMPLATES
        gen = ResponseGenerator()
        gen.add_template("custom_intent", "Custom response here.")
        self.assertIn("custom_intent", _TEMPLATES)
        self.assertIn("Custom response here.", _TEMPLATES["custom_intent"])

    def test_add_extra_template_to_existing_intent(self):
        from model_training.response_generator import ResponseGenerator, _TEMPLATES
        gen = ResponseGenerator()
        original_count = len(_TEMPLATES.get("greeting", []))
        gen.add_template("greeting", "Ahoy there!")
        self.assertGreater(len(_TEMPLATES["greeting"]), original_count)


class TestKBRetrieval(unittest.TestCase):
    """Tests for KB loading and retrieval with title + content indexing."""

    def _make_gen_with_kb(self, entries):
        import json
        import tempfile
        import os
        from model_training.response_generator import ResponseGenerator

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(entries, f)
            path = f.name

        gen = ResponseGenerator(knowledge_base_path=path, settings={"similarity_threshold": 0.01})
        os.unlink(path)
        return gen

    def test_kb_loaded_correctly(self):
        entries = [
            {"id": "1", "title": "Ibuprofen", "content": "Ibuprofen is an NSAID painkiller."},
        ]
        gen = self._make_gen_with_kb(entries)
        self.assertEqual(len(gen._knowledge_base), 1)

    def test_retrieve_returns_relevant_snippet(self):
        entries = [
            {"id": "1", "title": "Ibuprofen side effects", "content": "Common side effects include stomach upset and nausea."},
            {"id": "2", "title": "Vaccination schedule", "content": "Annual flu shots are recommended."},
        ]
        gen = self._make_gen_with_kb(entries)
        snippet = gen._retrieve("side effects of ibuprofen")
        self.assertIn("stomach", snippet)

    def test_title_included_in_index(self):
        # Searching by title keywords should match even if content is different
        entries = [
            {"id": "1", "title": "Metformin diabetes treatment", "content": "Used for type 2 diabetes."},
        ]
        gen = self._make_gen_with_kb(entries)
        snippet = gen._retrieve("metformin diabetes")
        self.assertGreater(len(snippet), 0)

    def test_empty_kb_returns_empty_snippet(self):
        from model_training.response_generator import ResponseGenerator
        gen = ResponseGenerator()
        snippet = gen._retrieve("any query")
        self.assertEqual(snippet, "")
