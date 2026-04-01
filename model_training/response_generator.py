"""
Response generator for pharmacy-connect chatbot.

Combines two strategies:
  1. **Template-based** – deterministic, hand-crafted responses for high-
     confidence intent matches.
  2. **NLU retrieval** – semantic similarity search over the knowledge base to
     surface relevant content snippets when no template matches.

When the caller provides a *confidence* score below
``LOW_CONFIDENCE_THRESHOLD`` the generator automatically selects the
``"low_confidence"`` template which asks the user to clarify their question,
rather than returning a potentially wrong answer.  Emergency and first-aid
intents are **never** re-routed on low confidence (patient safety).
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Confidence below which the generator asks for clarification.
# Emergency / first-aid intents are exempt from this threshold.
_LOW_CONFIDENCE_THRESHOLD = 0.40

# Intents that should never trigger clarification – safety-critical.
_SAFETY_INTENTS = frozenset(["emergency_assistance", "first_aid_guidance"])


# ── Built-in response templates ───────────────────────────────────────────────

_TEMPLATES: Dict[str, List[str]] = {
    "greeting": [
        "Hello! I'm the PharmacyConnect assistant. I can help with questions about "
        "medications, drug interactions, dosages, side effects, vaccinations, and pharmacies. "
        "What can I help you with?",
        "Hi there! I'm here to help with your pharmacy and health questions. "
        "Ask me about medications, prescriptions, side effects, or finding a pharmacy near you.",
    ],
    "farewell": [
        "Take care and stay healthy! Feel free to come back any time you have health questions.",
        "Goodbye! Remember — for urgent medical concerns always contact a healthcare "
        "professional or call 911.",
    ],
    "medication_inquiry": [
        "Here is what I found about that medication:\n\n{kb_snippet}\n\n"
        "💊 For personalized medication advice, always consult your pharmacist or physician.",
        "Here is relevant medication information:\n\n{kb_snippet}\n\n"
        "⚠️ This information is for educational purposes. "
        "Always follow your doctor's or pharmacist's instructions.",
    ],
    "drug_interaction_check": [
        "⚠️ Drug interactions can be serious and sometimes dangerous.\n\n{kb_snippet}\n\n"
        "Always consult your pharmacist or physician before combining medications — "
        "including over-the-counter drugs, supplements, and herbal remedies.",
        "Here is information about that drug interaction:\n\n{kb_snippet}\n\n"
        "🚨 If you are experiencing a severe reaction, call Poison Control at "
        "1-800-222-1222 or 911 immediately.",
    ],
    "side_effects_inquiry": [
        "Here are the side effects to be aware of:\n\n{kb_snippet}\n\n"
        "If you experience severe or unusual symptoms, stop taking the medication and "
        "contact your healthcare provider immediately.",
        "Possible side effects include:\n\n{kb_snippet}\n\n"
        "⚠️ Not everyone experiences side effects. Contact your doctor if symptoms "
        "are severe, persistent, or concerning.",
    ],
    "dosage_inquiry": [
        "Dosage guidance:\n\n{kb_snippet}\n\n"
        "⚠️ Always follow the dosage instructions on your prescription label or from "
        "your pharmacist. Never exceed the recommended dose without consulting a "
        "healthcare professional.",
        "Here is the dosing information:\n\n{kb_snippet}\n\n"
        "💊 For weight-based or age-specific dosing, ask your pharmacist for "
        "personalized guidance.",
    ],
    "vaccination_info": [
        "Vaccination information:\n\n{kb_snippet}\n\n"
        "💉 You can get vaccinated at your local pharmacy, clinic, or doctor's office. "
        "Ask your pharmacist about available vaccines and scheduling.",
        "Here is what I found about that vaccine:\n\n{kb_snippet}\n\n"
        "Visit vaccines.gov or contact your local pharmacy to check vaccine "
        "availability in your area.",
    ],
    "first_aid_guidance": [
        "🩹 First aid guidance:\n\n{kb_snippet}\n\n"
        "📞 For serious or worsening injuries, call 911 or go to the nearest "
        "emergency room immediately.",
        "Here is what to do:\n\n{kb_snippet}\n\n"
        "🚨 If the situation is life-threatening or you are unsure, call 911 right "
        "away. Do not delay emergency care.",
    ],
    "emergency_assistance": [
        "🚨 THIS SOUNDS LIKE AN EMERGENCY. Please call 911 immediately!\n\n"
        "For suspected poisoning or medication overdose, also call Poison Control: "
        "1-800-222-1222 (available 24/7).\n\n"
        "Additional guidance:\n\n{kb_snippet}",
        "🚨 Please call 911 or go to the nearest emergency room immediately!\n\n"
        "Poison Control (24/7): 1-800-222-1222\n\n"
        "{kb_snippet}",
    ],
    "pharmacy_location": [
        "To find a pharmacy near you:\n"
        "• Use Google Maps and search 'pharmacy near me'\n"
        "• Use your pharmacy chain's app (CVS, Walgreens, Rite Aid)\n"
        "• Call 1-800-PHARMACY for assistance\n\n"
        "Many pharmacies offer extended hours, drive-through service, and same-day "
        "prescription fulfillment.",
        "Here are ways to find a pharmacy near you:\n"
        "• Search 'open pharmacy near me' in your browser or maps app\n"
        "• Call your insurance company's member services line — they can direct you "
        "to in-network pharmacies\n"
        "• Use the CVS, Walgreens, or Rite Aid store locator on their websites",
    ],
    "prescription_refill": [
        "To refill your prescription, you have several options:\n"
        "• 📱 Use your pharmacy's mobile app (CVS, Walgreens, Rite Aid all have apps)\n"
        "• 🌐 Log in to your pharmacy's website\n"
        "• 📞 Call your pharmacy directly with your Rx number\n"
        "• 💊 Set up automatic refills so you never run out\n"
        "• 🏥 Ask your doctor's office to send a new prescription\n\n"
        "Most pharmacies allow refills 7–10 days before you run out.",
        "Prescription refill options:\n"
        "• Call your pharmacy and provide your prescription number\n"
        "• Use your pharmacy's app or website for online refills\n"
        "• Ask your doctor for a new prescription if you have no refills remaining\n\n"
        "💡 Tip: Sign up for auto-refills to never miss a dose.",
    ],
    "out_of_scope": [
        "I'm sorry, I'm specialized in pharmacy and healthcare topics and can't help "
        "with that. I can assist with: medications, drug interactions, dosages, side "
        "effects, vaccinations, first aid, prescription refills, and finding pharmacies.",
        "That question is outside my area of expertise. I'm a pharmacy assistant and "
        "can help with medication questions, drug interactions, dosages, side effects, "
        "and finding pharmacies near you. Is there a health or pharmacy question I can "
        "help with?",
    ],
    "general_health_question": [
        "Here's what I found on that topic:\n\n{kb_snippet}\n\n"
        "For personalized medical advice, please consult a healthcare provider or "
        "your pharmacist.",
        "Here is some relevant health information:\n\n{kb_snippet}\n\n"
        "💡 Your pharmacist is a great resource for general health and medication "
        "questions — and consultations are usually free!",
    ],
    "low_confidence": [
        "I want to make sure I understand your question correctly. Could you provide "
        "more details?\n\n"
        "For example, are you asking about:\n"
        "• A specific medication or drug?\n"
        "• Drug interactions or side effects?\n"
        "• A dosage or prescription question?\n"
        "• First aid or an emergency?\n"
        "• Finding a pharmacy?",
        "I'm not entirely sure I understood your question. Could you rephrase it or "
        "give me more context?\n\n"
        "I can help with medication information, drug interactions, dosages, side "
        "effects, vaccinations, first aid guidance, and pharmacy locations.",
    ],
}

_DEFAULT_FALLBACK = (
    "I'm sorry, I don't have enough information to answer that question. "
    "Please consult a pharmacist or healthcare provider."
)


class ResponseGenerator:
    """
    Generates natural-language responses for a given intent and query.

    Parameters
    ----------
    knowledge_base_path : str, optional
        Path to ``knowledge_base.json``.  When omitted the generator operates
        in template-only mode.
    settings : dict, optional
        Override default generation settings.
    """

    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        settings: Optional[Dict] = None,
    ):
        self.settings = settings or {}
        self.max_response_length: int = self.settings.get("max_response_length", 500)
        self.similarity_threshold: float = self.settings.get("similarity_threshold", 0.10)
        self.top_k: int = self.settings.get("top_k_responses", 3)

        self._knowledge_base: List[Dict] = []
        self._kb_texts: List[str] = []
        self._vectorizer = None
        self._kb_matrix = None

        if knowledge_base_path:
            self._load_knowledge_base(knowledge_base_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        intent: str,
        entities: Optional[Dict] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """
        Generate a response for *query* given the predicted *intent*.

        Parameters
        ----------
        query : str
            The user's raw input text.
        intent : str
            Predicted intent label.
        entities : dict, optional
            Named entities extracted from the query (e.g. ``{"medication": "ibuprofen"}``).
        confidence : float, optional
            ML model confidence in [0, 1].  When provided and below
            ``_LOW_CONFIDENCE_THRESHOLD``, a clarification response is returned
            instead of the normal template — unless *intent* is a safety-critical
            intent (``emergency_assistance`` or ``first_aid_guidance``).

        Returns
        -------
        str
            The generated response text.
        """
        entities = entities or {}

        # Route low-confidence predictions to clarification (safety intents exempt).
        effective_intent = intent
        if (
            confidence is not None
            and confidence < _LOW_CONFIDENCE_THRESHOLD
            and intent not in _SAFETY_INTENTS
        ):
            effective_intent = "low_confidence"

        # 1. Retrieve relevant KB snippet (if KB is loaded)
        kb_snippet = self._retrieve(query) if self._knowledge_base else ""

        # 2. Select template
        template = self._select_template(effective_intent)

        # 3. Fill template
        response = self._fill_template(template, entities, kb_snippet)

        # 4. Truncate if necessary
        if len(response) > self.max_response_length:
            response = response[: self.max_response_length].rsplit(" ", 1)[0] + "…"

        return response

    def add_template(self, intent: str, template: str) -> None:
        """Register a custom response template for *intent*."""
        if intent not in _TEMPLATES:
            _TEMPLATES[intent] = []
        _TEMPLATES[intent].append(template)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_knowledge_base(self, path: str) -> None:
        """Load and index the knowledge base from a JSON file."""
        if not os.path.isfile(path):
            logger.warning("Knowledge base not found at %s – using template-only mode.", path)
            return

        with open(path, encoding="utf-8") as f:
            self._knowledge_base = json.load(f)

        # Index both title and content for richer semantic matching.
        self._kb_texts = [
            (entry.get("title", "") + " " + entry.get("content", "")).strip()
            for entry in self._knowledge_base
        ]
        self._build_index()
        logger.info("Knowledge base loaded: %d entries", len(self._knowledge_base))

    def _build_index(self) -> None:
        """Build a TF-IDF index over the knowledge base texts."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_features=20_000, sublinear_tf=True
            )
            self._kb_matrix = self._vectorizer.fit_transform(self._kb_texts)
        except ImportError:
            logger.warning("scikit-learn not installed – KB retrieval disabled.")

    def _retrieve(self, query: str) -> str:
        """Return the most relevant KB snippet for *query*."""
        if self._vectorizer is None or self._kb_matrix is None:
            return ""

        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            query_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(query_vec, self._kb_matrix)[0]
            top_indices = np.argsort(sims)[::-1][: self.top_k]

            snippets = []
            for idx in top_indices:
                if sims[idx] >= self.similarity_threshold:
                    # Use the original content field for the response snippet.
                    entry = self._knowledge_base[idx]
                    content = entry.get("content", self._kb_texts[idx])
                    snippets.append(content[:300])

            return "\n\n".join(snippets) if snippets else ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("KB retrieval failed: %s", exc)
            return ""

    @staticmethod
    def _select_template(intent: str) -> str:
        """Pick the first template for *intent* (or the fallback)."""
        templates = _TEMPLATES.get(intent)
        if templates:
            return templates[0]
        return _DEFAULT_FALLBACK

    @staticmethod
    def _fill_template(template: str, entities: Dict, kb_snippet: str) -> str:
        """Replace placeholders in *template* with entity values / KB snippet."""
        result = template

        # Fill {kb_snippet}
        if "{kb_snippet}" in result:
            result = result.replace("{kb_snippet}", kb_snippet or "No relevant information found.")

        # Fill named entities
        for key, value in entities.items():
            result = result.replace("{" + key + "}", str(value))

        # Remove unfilled placeholders, including any preceding preposition/article
        # so we don't leave dangling phrases like "about , I recommend…".
        result = re.sub(r"\s*\b(?:about|for|of|on|with|regarding)\s+\{[^}]+\}", "", result)
        result = re.sub(r"\{[^}]+\}", "", result)
        return result.strip()

