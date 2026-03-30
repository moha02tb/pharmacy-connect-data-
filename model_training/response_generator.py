"""
Response generator for pharmacy-connect chatbot.

Combines two strategies:
  1. **Template-based** – deterministic, hand-crafted responses for high-
     confidence intent matches.
  2. **NLU retrieval** – semantic similarity search over the knowledge base to
     surface relevant content snippets when no template matches.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Built-in response templates ───────────────────────────────────────────────

_TEMPLATES: Dict[str, List[str]] = {
    "greeting": [
        "Hello! I'm PharmacyConnect assistant. How can I help you today?",
        "Hi there! I can answer questions about medications, pharmacies, and health. What do you need?",
    ],
    "farewell": [
        "Take care and stay healthy!",
        "Goodbye! Don't hesitate to return if you have more health questions.",
    ],
    "medication_inquiry": [
        "For information about {medication}, I recommend consulting your pharmacist or physician. "
        "Here's what our knowledge base says:\n\n{kb_snippet}",
        "Here is relevant medication information:\n\n{kb_snippet}",
    ],
    "drug_interaction_check": [
        "Drug interactions can be serious. Here's relevant information:\n\n{kb_snippet}\n\n"
        "⚠️ Always consult your pharmacist before combining medications.",
    ],
    "side_effects_inquiry": [
        "Possible side effects to be aware of:\n\n{kb_snippet}\n\n"
        "If you experience severe symptoms, contact your healthcare provider immediately.",
    ],
    "dosage_inquiry": [
        "Dosage guidance:\n\n{kb_snippet}\n\n"
        "⚠️ Always follow your prescription label or ask your pharmacist for personalised advice.",
    ],
    "vaccination_info": [
        "Vaccination information:\n\n{kb_snippet}\n\n"
        "Visit your local pharmacy or clinic to schedule a vaccination appointment.",
    ],
    "first_aid_guidance": [
        "First aid guidance:\n\n{kb_snippet}\n\n"
        "📞 For serious emergencies, call 911 immediately.",
    ],
    "emergency_assistance": [
        "🚨 This sounds like an emergency. Please call 911 or Poison Control (1-800-222-1222) immediately.\n\n"
        "Additional information:\n\n{kb_snippet}",
    ],
    "pharmacy_location": [
        "To find a pharmacy near you, please use your pharmacy app or visit your pharmacy's website. "
        "You can also call 1-800-PHARMACY for assistance.",
    ],
    "prescription_refill": [
        "To refill your prescription you can:\n"
        "• Call your pharmacy directly\n"
        "• Use your pharmacy's mobile app\n"
        "• Ask your doctor to send a new prescription\n\n"
        "Do you need help with anything else?",
    ],
    "out_of_scope": [
        "I'm sorry, I can only assist with pharmacy and health-related questions. "
        "Please consult a healthcare professional for other queries.",
    ],
    "general_health_question": [
        "Here's what I found on that topic:\n\n{kb_snippet}\n\n"
        "For personalised advice, please consult a healthcare provider.",
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
        self.similarity_threshold: float = self.settings.get("similarity_threshold", 0.05)
        self.top_k: int = self.settings.get("top_k_responses", 3)

        self._knowledge_base: List[Dict] = []
        self._kb_texts: List[str] = []
        self._vectorizer = None
        self._kb_matrix = None

        if knowledge_base_path:
            self._load_knowledge_base(knowledge_base_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, query: str, intent: str, entities: Optional[Dict] = None) -> str:
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

        Returns
        -------
        str
            The generated response text.
        """
        entities = entities or {}

        # 1. Retrieve relevant KB snippet (if KB is loaded)
        kb_snippet = self._retrieve(query) if self._knowledge_base else ""

        # 2. Select template
        template = self._select_template(intent)

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

        self._kb_texts = [entry.get("content", "") for entry in self._knowledge_base]
        self._build_index()
        logger.info("Knowledge base loaded: %d entries", len(self._knowledge_base))

    def _build_index(self) -> None:
        """Build a TF-IDF index over the knowledge base texts."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)
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
                    content = self._kb_texts[idx]
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

        # Remove any unfilled placeholders
        result = re.sub(r"\{[^}]+\}", "", result)
        return result.strip()
