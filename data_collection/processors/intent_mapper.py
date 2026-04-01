"""
intent_mapper.py – Map medical Q&A pairs to the 13 pharmacy intent labels.

Intent labels (from ``data_collection/config.py``):
    medication_inquiry, drug_interaction_check, side_effects_inquiry,
    dosage_inquiry, pharmacy_location, prescription_refill, vaccination_info,
    first_aid_guidance, emergency_assistance, general_health_question,
    greeting, farewell, out_of_scope
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Keyword → intent mapping ──────────────────────────────────────────────────
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "medication_inquiry": [
        "medication",
        "medicine",
        "drug",
        "prescription",
        "pharmaceutical",
        "pill",
        "tablet",
        "capsule",
        "antibiotic",
        "painkiller",
        "generic",
        "brand name",
        "what is this drug",
        "what medication",
        "what medicine",
        "tell me about",
        "what is used for",
        "what does it treat",
        "what conditions",
        "is it a",
        "anti-inflammatory",
        "blood thinner",
        "antidepressant",
        "statin",
        "beta blocker",
        "antihistamine",
    ],
    "drug_interaction_check": [
        "interaction",
        "drug interaction",
        "contraindicated",
        "combine",
        "take together",
        "mix with",
        "can i take",
        "safe with",
        "combination",
        "interfere",
        "interact",
        "is it ok to take",
        "affect",
        "grapefruit",
        "alcohol with",
        "coffee with",
        "food with",
    ],
    "side_effects_inquiry": [
        "side effect",
        "adverse",
        "reaction",
        "symptoms after taking",
        "side-effect",
        "after taking",
        "causes",
        "side effects of",
        "negative effect",
        "bad effect",
        "risk",
        "complication",
        "warning",
        "can it cause",
        "does it cause",
        "weight gain",
        "hair loss",
        "nausea",
        "dizziness",
        "dry cough",
    ],
    "dosage_inquiry": [
        "dosage",
        "dose",
        "how much",
        "how to take",
        "instructions",
        "how many",
        "how often",
        "frequency",
        "when to take",
        "maximum dose",
        "overdose",
        "mg",
        "milligram",
        "twice a day",
        "once a day",
        "daily dose",
        "per day",
        "by weight",
        "for adults",
        "for children",
        "too much",
    ],
    "pharmacy_location": [
        "pharmacy",
        "pharmacist",
        "drugstore",
        "near me",
        "where can i",
        "find a pharmacy",
        "closest pharmacy",
        "open pharmacy",
        "pharmacy hours",
        "cvs",
        "walgreens",
        "rite aid",
        "24-hour pharmacy",
        "open now",
        "open today",
        "pharmacy open",
        "deliver medications",
        "accepts my insurance",
        "transfer prescription",
    ],
    "prescription_refill": [
        "refill",
        "renew prescription",
        "reorder",
        "running out",
        "need more",
        "prescription expired",
        "auto refill",
        "ran out",
        "reorder medication",
        "refills remaining",
        "automatic refill",
        "early refill",
        "prescription number",
        "new prescription",
        "out of medication",
    ],
    "vaccination_info": [
        "vaccine",
        "vaccination",
        "immunization",
        "booster",
        "flu shot",
        "covid vaccine",
        "immunize",
        "shot",
        "inoculation",
        "mmr",
        "tdap",
        "tetanus",
        "shingles",
        "pneumonia vaccine",
        "hpv vaccine",
        "travel vaccine",
        "childhood vaccine",
        "adult vaccine",
    ],
    "first_aid_guidance": [
        "first aid",
        "bleeding",
        "wound",
        "burn",
        "cpr",
        "aed",
        "cut",
        "injury",
        "bandage",
        "sprain",
        "fracture",
        "choking",
        "seizure",
        "bee sting",
        "scald",
        "minor injury",
        "how to treat",
        "what to do for",
    ],
    "emergency_assistance": [
        "emergency",
        "call 911",
        "poison control",
        "overdose",
        "urgent",
        "life threatening",
        "ambulance",
        "critical",
        "heart attack",
        "not waking up",
        "not responding",
        "collapsed",
        "swallowed",
        "allergic reaction",
        "anaphylaxis",
        "unconscious",
        "not breathing",
    ],
    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        "howdy",
        "what's up",
        "good day",
        "morning",
    ],
    "farewell": [
        "goodbye",
        "bye",
        "see you",
        "thank you",
        "thanks",
        "have a good day",
        "farewell",
        "all done",
        "take care",
        "that's all",
        "i'm done",
    ],
}

# Minimum confidence to label a record (0–1 scale)
_DEFAULT_CONFIDENCE_THRESHOLD = 0.3


class IntentMapper:
    """
    Map medical Q&A text to one of 13 pharmacy intent labels.

    Parameters
    ----------
    confidence_threshold:
        Minimum keyword-match ratio required to assign a non-default intent.
        Records below this threshold are labelled ``"general_health_question"``.
    """

    def __init__(self, confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = confidence_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def map(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Add ``intent`` and ``confidence`` fields to each Q&A pair in *qa_pairs*.

        Each input dict should have at least a ``"question"`` key.  The intent
        is inferred from the question text.

        Returns the same list with ``intent`` and ``confidence`` fields added
        in-place.
        """
        for pair in qa_pairs:
            intent, confidence = self.predict(pair.get("question", ""))
            pair["intent"] = intent
            pair["confidence"] = confidence
        return qa_pairs

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict the intent label and confidence for *text*.

        Returns
        -------
        (intent, confidence)
            *intent* is a string label; *confidence* is a float in [0, 1].
        """
        lower = text.lower()
        scores: Dict[str, int] = {}

        for intent, keywords in _INTENT_KEYWORDS.items():
            matches = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", lower))
            if matches:
                scores[intent] = matches

        if not scores:
            return "general_health_question", 0.0

        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]
        total_keywords = len(_INTENT_KEYWORDS[best_intent])
        confidence = min(best_score / max(total_keywords, 1), 1.0)

        if confidence < self.confidence_threshold:
            return "general_health_question", confidence

        return best_intent, confidence

    def filter_by_confidence(
        self,
        qa_pairs: List[Dict],
        min_confidence: Optional[float] = None,
    ) -> List[Dict]:
        """Return only pairs whose ``confidence`` meets *min_confidence*."""
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold
        return [p for p in qa_pairs if p.get("confidence", 0.0) >= threshold]
