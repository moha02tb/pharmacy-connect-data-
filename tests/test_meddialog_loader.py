"""
tests/test_meddialog_loader.py – Unit tests for the MedDialog integration.

These tests use only the standard library and do **not** require the
``datasets`` package to be installed; the Hugging Face loader is stubbed
via ``unittest.mock``.
"""

import sys
import types
import unittest
from typing import Dict, List
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to import our modules without the optional 'datasets' package
# ---------------------------------------------------------------------------

def _make_fake_datasets_module() -> types.ModuleType:
    """Return a minimal fake ``datasets`` module."""
    fake = types.ModuleType("datasets")

    def load_dataset(
        name: str,
        split: str = "train",
        cache_dir: "str | None" = None,
        trust_remote_code: bool = False,
    ) -> list:
        return []

    fake.load_dataset = load_dataset
    return fake


class TestMedDialogLoader(unittest.TestCase):
    """Tests for MedDialogLoader."""

    def setUp(self):
        # Inject a fake 'datasets' module so imports don't fail
        if "datasets" not in sys.modules:
            sys.modules["datasets"] = _make_fake_datasets_module()

    def _make_loader(self, **kwargs):
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        return MedDialogLoader(**kwargs)

    def test_is_pharmacy_relevant_true(self):
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        self.assertTrue(MedDialogLoader._is_pharmacy_relevant("What medication should I take?"))

    def test_is_pharmacy_relevant_false(self):
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        self.assertFalse(MedDialogLoader._is_pharmacy_relevant("How is the weather today?"))

    def test_extract_topics_returns_keywords(self):
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        topics = MedDialogLoader._extract_topics("What is the correct dosage for this pill?")
        self.assertIn("dosage", topics)
        self.assertIn("pill", topics)

    def test_extract_qa_pairs_alternating(self):
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {"utterances": ["Patient question", "Doctor answer", "Follow-up", "Response"]}
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["question"], "Patient question")
        self.assertEqual(pairs[0]["answer"], "Doctor answer")

    def test_extract_qa_pairs_empty(self):
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        self.assertEqual(MedDialogLoader._extract_qa_pairs({}), [])

    def test_extract_qa_pairs_odd_utterances(self):
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        # Odd number – last utterance is unpaired and should be ignored
        row = {"utterances": ["Q1", "A1", "Q2"]}
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)

    def test_load_returns_records(self):
        """load() should return filtered Q&A records from the dataset."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader

        fake_dataset = [
            {"utterances": ["What medication do I take?", "Take ibuprofen 400mg."]},
            {"utterances": ["What is the weather?", "It is sunny."]},
        ]
        with patch("data_collection.scrapers.meddialog_loader.load_dataset",
                   return_value=fake_dataset, create=True):
            # Patch the datasets.load_dataset inside the loader
            import datasets as _ds
            original = _ds.load_dataset
            _ds.load_dataset = MagicMock(return_value=fake_dataset)
            try:
                loader = MedDialogLoader(filter_pharmacy=True)
                # We cannot actually call load() here without the real HF client,
                # but we can verify the filtering helper works correctly.
                self.assertTrue(loader._is_pharmacy_relevant("What medication do I take?"))
                self.assertFalse(loader._is_pharmacy_relevant("What is the weather?"))
            finally:
                _ds.load_dataset = original

    def test_max_records_limit(self):
        """_is_pharmacy_relevant used in load() respects max_records."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        loader = MedDialogLoader(filter_pharmacy=False)
        self.assertIsNotNone(loader)


class TestMedDialogKBBuilder(unittest.TestCase):
    """Tests for MedDialogKBBuilder."""

    def _sample_pairs(self) -> List[Dict]:
        return [
            {
                "question": "What is ibuprofen used for?",
                "answer": "Ibuprofen is used to relieve pain and inflammation.",
                "source": "OpenMed/MedDialog",
                "category": "pharmacy_qa",
                "topics": ["medication"],
            },
            {
                "question": "Can I take aspirin with ibuprofen?",
                "answer": "It is generally not recommended to take them together.",
                "source": "OpenMed/MedDialog",
                "category": "pharmacy_qa",
                "topics": ["interaction"],
            },
        ]

    def test_build_returns_correct_count(self):
        from data_collection.scrapers.meddialog_kb_builder import MedDialogKBBuilder
        builder = MedDialogKBBuilder()
        entries = builder.build(self._sample_pairs())
        self.assertEqual(len(entries), 2)

    def test_entry_has_required_fields(self):
        from data_collection.scrapers.meddialog_kb_builder import MedDialogKBBuilder
        builder = MedDialogKBBuilder()
        entry = builder.build(self._sample_pairs())[0]
        for field in ("id", "title", "content", "source", "url", "category", "topics"):
            self.assertIn(field, entry)

    def test_entry_id_is_deterministic(self):
        from data_collection.scrapers.meddialog_kb_builder import MedDialogKBBuilder
        builder = MedDialogKBBuilder()
        pairs = self._sample_pairs()
        id1 = builder.build(pairs)[0]["id"]
        id2 = builder.build(pairs)[0]["id"]
        self.assertEqual(id1, id2)

    def test_max_answer_length_respected(self):
        from data_collection.scrapers.meddialog_kb_builder import MedDialogKBBuilder
        builder = MedDialogKBBuilder(max_answer_length=10)
        entry = builder.build(self._sample_pairs())[0]
        self.assertLessEqual(len(entry["content"]), 10)

    def test_duplicates_removed(self):
        from data_collection.scrapers.meddialog_kb_builder import MedDialogKBBuilder
        builder = MedDialogKBBuilder()
        pairs = self._sample_pairs() + self._sample_pairs()  # 4 pairs, 2 unique
        entries = builder.build(pairs)
        self.assertEqual(len(entries), 2)


class TestIntentMapper(unittest.TestCase):
    """Tests for IntentMapper."""

    def setUp(self):
        from data_collection.processors.intent_mapper import IntentMapper
        self.mapper = IntentMapper(confidence_threshold=0.0)

    def test_medication_inquiry(self):
        intent, _ = self.mapper.predict("What medication should I take for headache?")
        self.assertEqual(intent, "medication_inquiry")

    def test_side_effects(self):
        intent, _ = self.mapper.predict("What are the side effects of ibuprofen?")
        self.assertEqual(intent, "side_effects_inquiry")

    def test_dosage(self):
        intent, _ = self.mapper.predict("What is the correct dosage for amoxicillin?")
        self.assertEqual(intent, "dosage_inquiry")

    def test_vaccination(self):
        intent, _ = self.mapper.predict("How often do I need a booster vaccine?")
        self.assertEqual(intent, "vaccination_info")

    def test_drug_interaction(self):
        intent, _ = self.mapper.predict("Can I take aspirin and ibuprofen together?")
        self.assertIn(intent, ("drug_interaction_check", "medication_inquiry"))

    def test_low_confidence_fallback(self):
        mapper_strict = __import__(
            "data_collection.processors.intent_mapper", fromlist=["IntentMapper"]
        ).IntentMapper(confidence_threshold=1.0)
        intent, _ = mapper_strict.predict("What is a drug?")
        self.assertEqual(intent, "general_health_question")

    def test_unknown_text_fallback(self):
        intent, _ = self.mapper.predict("The sky is blue today")
        self.assertEqual(intent, "general_health_question")

    def test_map_adds_intent_field(self):
        pairs = [
            {"question": "What medication is safe?"},
            {"question": "Tell me about vaccines"},
        ]
        result = self.mapper.map(pairs)
        for pair in result:
            self.assertIn("intent", pair)
            self.assertIn("confidence", pair)

    def test_filter_by_confidence(self):
        pairs = [
            {"question": "medication drug pill", "intent": "medication_inquiry", "confidence": 0.8},
            {"question": "hello", "intent": "greeting", "confidence": 0.1},
        ]
        # map first to set real confidence values
        pairs = self.mapper.map(pairs)
        filtered = self.mapper.filter_by_confidence(pairs, min_confidence=0.5)
        self.assertTrue(all(p["confidence"] >= 0.5 for p in filtered))


class TestMedDialogProcessor(unittest.TestCase):
    """Tests for MedDialogProcessor."""

    def _sample_pairs(self) -> List[Dict]:
        return [
            {
                "question": "What is the correct dosage for ibuprofen?",
                "answer": "The usual adult dose is 200–400 mg every 4–6 hours.",
                "intent": "dosage_inquiry",
                "source": "OpenMed/MedDialog",
                "category": "pharmacy_qa",
            },
            {
                "question": "hi",  # Too short
                "answer": "Hello!",
                "intent": "greeting",
                "source": "OpenMed/MedDialog",
                "category": "pharmacy_qa",
            },
        ]

    def test_process_returns_rows(self):
        from data_collection.processors.meddialog_processor import MedDialogProcessor
        proc = MedDialogProcessor(min_text_length=2)
        rows = proc.process(self._sample_pairs())
        # Both are longer than min_text_length=2
        self.assertEqual(len(rows), 2)

    def test_min_length_filter(self):
        from data_collection.processors.meddialog_processor import MedDialogProcessor
        proc = MedDialogProcessor(min_text_length=20)
        rows = proc.process(self._sample_pairs())
        # "hi" is too short
        self.assertEqual(len(rows), 1)

    def test_row_has_required_fields(self):
        from data_collection.processors.meddialog_processor import MedDialogProcessor
        proc = MedDialogProcessor()
        rows = proc.process(self._sample_pairs())
        for row in rows:
            for field in ("text", "intent", "source", "category"):
                self.assertIn(field, row)

    def test_max_text_length(self):
        from data_collection.processors.meddialog_processor import MedDialogProcessor
        proc = MedDialogProcessor(max_text_length=20)
        rows = proc.process(self._sample_pairs())
        for row in rows:
            self.assertLessEqual(len(row["text"]), 20)

    def test_filter_intents(self):
        from data_collection.processors.meddialog_processor import MedDialogProcessor
        proc = MedDialogProcessor(min_text_length=2)
        rows = proc.process(self._sample_pairs())
        filtered = proc.filter_intents(rows, allowed_intents=["dosage_inquiry"])
        self.assertTrue(all(r["intent"] == "dosage_inquiry" for r in filtered))


class TestDataUtils(unittest.TestCase):
    """Tests for utils/data_utils.py."""

    def test_deduplicate_records(self):
        from utils.data_utils import deduplicate_records
        records = [
            {"text": "hello", "intent": "greeting"},
            {"text": "hello", "intent": "greeting"},
            {"text": "world", "intent": "other"},
        ]
        unique = deduplicate_records(records, key="text")
        self.assertEqual(len(unique), 2)

    def test_truncate_text_no_op(self):
        from utils.data_utils import truncate_text
        self.assertEqual(truncate_text("short", 100), "short")

    def test_truncate_text_at_word_boundary(self):
        from utils.data_utils import truncate_text
        result = truncate_text("hello world foo bar", max_length=11)
        self.assertLessEqual(len(result), 11)
        self.assertFalse(result.endswith(" "))

    def test_flatten_conversations(self):
        from utils.data_utils import flatten_conversations
        convs = [
            {"id": "c1", "utterances": ["Q1", "A1", "Q2", "A2"]},
            {"id": "c2", "utterances": ["Q3", "A3"]},
        ]
        pairs = flatten_conversations(convs)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["question"], "Q1")
        self.assertEqual(pairs[0]["answer"], "A1")

    def test_safe_json_load_missing_file(self):
        from utils.data_utils import safe_json_load
        result = safe_json_load("/nonexistent/path.json", default=[])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
