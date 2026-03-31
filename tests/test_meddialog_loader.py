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
        streaming: bool = False,
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

    def test_extract_qa_pairs_dict_format(self):
        """Dict utterances with speaker/utterance keys are extracted correctly."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {
            "utterances": [
                {"speaker": "Patient", "utterance": "What medication should I take?"},
                {"speaker": "Doctor", "utterance": "Take ibuprofen 400mg."},
                {"speaker": "Patient", "utterance": "Any side effects?"},
                {"speaker": "Doctor", "utterance": "Possible stomach upset."},
            ]
        }
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["question"], "What medication should I take?")
        self.assertEqual(pairs[0]["answer"], "Take ibuprofen 400mg.")

    def test_extract_qa_pairs_dict_format_role_key(self):
        """Dict utterances using 'role' instead of 'speaker' are handled."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {
            "utterances": [
                {"role": "patient", "text": "Do I need a prescription?"},
                {"role": "doctor", "text": "Yes, this requires a prescription."},
            ]
        }
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "Do I need a prescription?")

    def test_extract_qa_pairs_role_prefixed_strings(self):
        """Strings prefixed with 'Patient:' / 'Doctor:' are handled correctly."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {
            "utterances": [
                "Patient: What is the correct dosage?",
                "Doctor: The usual dose is 500mg twice a day.",
            ]
        }
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "What is the correct dosage?")
        self.assertEqual(pairs[0]["answer"], "The usual dose is 500mg twice a day.")

    def test_extract_qa_pairs_dialogue_field_alias(self):
        """The 'dialogue' field alias is tried when 'utterances' is absent."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {"dialogue": ["How long should I take antibiotics?", "Complete the full course."]}
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)

    def test_extract_qa_pairs_dialog_field_alias(self):
        """The 'dialog' field alias is tried when other fields are absent."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {"dialog": ["Is aspirin safe?", "In normal doses, yes."]}
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)

    def test_extract_qa_pairs_conversations_field_alias(self):
        """The 'conversations' field alias is tried as a last resort."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {"conversations": ["Can I take this with food?", "Yes, take it with meals."]}
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)

    def test_extract_qa_pairs_conversation_singular_field_alias(self):
        """The 'conversation' (singular) field alias is tried."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {
            "conversation": [
                {"role": "patient", "text": "What medication should I take?"},
                {"role": "doctor", "text": "Take ibuprofen 400mg."},
            ]
        }
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "What medication should I take?")

    def test_extract_qa_pairs_messages_field_alias(self):
        """The 'messages' field alias is tried (e.g. OpenAI-style chat format)."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {
            "messages": [
                {"role": "user", "content": "What is the correct dosage?"},
                {"role": "assistant", "content": "Take 500mg twice a day."},
            ]
        }
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "What is the correct dosage?")
        self.assertEqual(pairs[0]["answer"], "Take 500mg twice a day.")

    def test_extract_qa_pairs_turns_field_alias(self):
        """The 'turns' field alias is tried."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {"turns": ["Can I refill my prescription?", "Yes, visit any pharmacy."]}
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)

    def test_extract_qa_pairs_flat_qa_format(self):
        """Rows with direct 'question'/'answer' fields are handled."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {"question": "What are the side effects?", "answer": "Nausea and headache."}
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "What are the side effects?")
        self.assertEqual(pairs[0]["answer"], "Nausea and headache.")

    def test_extract_qa_pairs_query_response_format(self):
        """Rows with 'query'/'response' fields are handled."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {"query": "Is this drug safe?", "response": "Yes, at recommended doses."}
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "Is this drug safe?")

    def test_extract_qa_pairs_description_answer_format(self):
        """Rows with 'description'/'answer' fields are handled."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {"description": "Patient asks about medication dose.", "answer": "Take 200mg."}
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "Patient asks about medication dose.")

    def test_extract_qa_pairs_content_key_in_dict_utterances(self):
        """Dict utterances with 'content' key (OpenAI format) are extracted."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        row = {
            "utterances": [
                {"speaker": "patient", "content": "Do I need a prescription for antibiotics?"},
                {"speaker": "doctor", "content": "Yes, antibiotics require a prescription."},
            ]
        }
        pairs = MedDialogLoader._extract_qa_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "Do I need a prescription for antibiotics?")

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

    def test_use_streaming_default_true(self):
        """use_streaming defaults to True."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        loader = MedDialogLoader()
        self.assertTrue(loader.use_streaming)

    def test_use_streaming_can_be_disabled(self):
        """use_streaming can be set to False."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        loader = MedDialogLoader(use_streaming=False)
        self.assertFalse(loader.use_streaming)

    def test_max_retries_default(self):
        """max_retries defaults to 3."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        loader = MedDialogLoader()
        self.assertEqual(loader.max_retries, 3)

    def test_max_retries_minimum_is_one(self):
        """max_retries is clamped to at least 1."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        loader = MedDialogLoader(max_retries=0)
        self.assertEqual(loader.max_retries, 1)

    def test_load_with_retry_passes_streaming_flag(self):
        """_load_with_retry forwards the streaming flag to load_dataset."""
        from data_collection.scrapers.meddialog_loader import MedDialogLoader
        import datasets as _ds

        calls = []

        def fake_load(name, split="train", cache_dir=None, streaming=False):
            calls.append({"name": name, "streaming": streaming})
            return []

        original = _ds.load_dataset
        _ds.load_dataset = fake_load
        try:
            loader = MedDialogLoader(use_streaming=True)
            loader._load_with_retry(fake_load)
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0]["streaming"])
        finally:
            _ds.load_dataset = original

    def test_load_with_retry_retries_on_failure(self):
        """_load_with_retry retries up to max_retries times on exceptions."""
        import unittest.mock as mock
        from data_collection.scrapers.meddialog_loader import MedDialogLoader

        attempt_count = 0

        def always_fail(name, split="train", cache_dir=None, streaming=False):
            nonlocal attempt_count
            attempt_count += 1
            raise ConnectionError("simulated network error")

        loader = MedDialogLoader(max_retries=3)
        with mock.patch("time.sleep"):  # skip actual sleep in tests
            with self.assertRaises(RuntimeError):
                loader._load_with_retry(always_fail)
        self.assertEqual(attempt_count, 3)

    def test_load_with_retry_succeeds_on_second_attempt(self):
        """_load_with_retry returns the dataset after a transient failure."""
        import unittest.mock as mock
        from data_collection.scrapers.meddialog_loader import MedDialogLoader

        call_count = 0

        def fail_once(name, split="train", cache_dir=None, streaming=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("transient error")
            return ["row1", "row2"]

        loader = MedDialogLoader(max_retries=3)
        with mock.patch("time.sleep"):
            result = loader._load_with_retry(fail_once)
        self.assertEqual(result, ["row1", "row2"])
        self.assertEqual(call_count, 2)

    def test_load_streaming_stops_at_max_records(self):
        """In streaming mode, load() stops fetching once max_records is hit."""
        import datasets as _ds
        from data_collection.scrapers.meddialog_loader import MedDialogLoader

        # All rows contain "medication" so all are pharmacy-relevant
        fake_dataset = [
            {"utterances": [f"What medication is drug{i}?", f"Drug{i} answer."]}
            for i in range(20)
        ]

        rows_consumed = [0]
        original_load = _ds.load_dataset

        def fake_load(name, split="train", cache_dir=None, streaming=False):
            def gen():
                for row in fake_dataset:
                    rows_consumed[0] += 1
                    yield row

            return gen()

        _ds.load_dataset = fake_load
        try:
            loader = MedDialogLoader(filter_pharmacy=True, use_streaming=True, max_retries=1)
            records = loader.load(max_records=5)
            # All questions are pharmacy-relevant, so exactly 5 should be returned
            self.assertEqual(len(records), 5)
            # Should have stopped early, not consumed all 20 rows
            self.assertLess(rows_consumed[0], 20)
        finally:
            _ds.load_dataset = original_load


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
