"""
Data augmentation utilities for expanding the training dataset.

Techniques implemented:
  - Synonym replacement  (requires NLTK WordNet)
  - Random deletion
  - Random swap
  - Paraphrase-style template expansion
"""

import logging
import random
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DataAugmenter:
    """
    Generates synthetic training samples from existing labelled data.

    Parameters
    ----------
    settings : dict, optional
        Override default augmentation hyper-parameters.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, settings: Optional[Dict] = None, seed: int = 42):
        self.settings = settings or {}
        self.augmentation_factor: int = self.settings.get("augmentation_factor", 3)
        self.synonym_prob: float = self.settings.get("synonym_replacement_prob", 0.2)
        self.deletion_prob: float = self.settings.get("random_deletion_prob", 0.1)
        self.swap_prob: float = self.settings.get("random_swap_prob", 0.1)
        random.seed(seed)

        # Lazy-load NLTK WordNet for synonym replacement
        self._wordnet = None

    # ── Public API ────────────────────────────────────────────────────────────

    def augment(self, training_data: List[Dict]) -> List[Dict]:
        """
        Augment *training_data* by generating synthetic samples.

        Parameters
        ----------
        training_data : list[dict]
            Each entry must have ``"text"`` and ``"intent"`` keys.

        Returns
        -------
        list[dict]
            The original data plus all generated samples.
        """
        if not training_data:
            return training_data

        augmented: List[Dict] = list(training_data)

        for record in training_data:
            text = record["text"]
            intent = record["intent"]

            for _ in range(self.augmentation_factor - 1):
                new_text = self._augment_text(text)
                if new_text and new_text != text:
                    augmented.append({"text": new_text, "intent": intent})

        logger.info(
            "Augmented dataset: %d → %d samples", len(training_data), len(augmented)
        )
        return augmented

    # ── Augmentation operations ───────────────────────────────────────────────

    def _augment_text(self, text: str) -> str:
        """Apply a random combination of augmentation operations to *text*."""
        words = text.split()
        if len(words) < 3:
            return text

        ops = []
        if random.random() < self.synonym_prob:
            ops.append("synonym")
        if random.random() < self.deletion_prob:
            ops.append("delete")
        if random.random() < self.swap_prob:
            ops.append("swap")

        for op in ops:
            if op == "synonym":
                words = self._synonym_replacement(words)
            elif op == "delete":
                words = self._random_deletion(words)
            elif op == "swap":
                words = self._random_swap(words)

        return " ".join(words).strip()

    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """Replace a random word with one of its synonyms."""
        wordnet = self._get_wordnet()
        if not wordnet:
            return words

        result = list(words)
        indices = list(range(len(result)))
        random.shuffle(indices)

        for idx in indices:
            word = result[idx]
            synonyms = self._get_synonyms(word, wordnet)
            if synonyms:
                result[idx] = random.choice(synonyms)
                break

        return result

    @staticmethod
    def _random_deletion(words: List[str], min_words: int = 2) -> List[str]:
        """Randomly delete one non-important word, keeping at least *min_words*."""
        if len(words) <= min_words:
            return words

        # Avoid deleting the first word (often carries intent signal)
        idx = random.randint(1, len(words) - 1)
        return words[:idx] + words[idx + 1 :]

    @staticmethod
    def _random_swap(words: List[str]) -> List[str]:
        """Randomly swap two adjacent words."""
        if len(words) < 2:
            return words

        result = list(words)
        idx = random.randint(0, len(result) - 2)
        result[idx], result[idx + 1] = result[idx + 1], result[idx]
        return result

    # ── NLTK helpers ──────────────────────────────────────────────────────────

    def _get_wordnet(self):
        """Lazily load NLTK WordNet; returns ``None`` if NLTK is unavailable."""
        if self._wordnet is not None:
            return self._wordnet
        try:
            import nltk
            from nltk.corpus import wordnet

            # Download silently if needed
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)

            self._wordnet = wordnet
        except ImportError:
            logger.debug("NLTK not installed – synonym replacement disabled.")
            self._wordnet = False  # Sentinel: already tried, skip in future
        return self._wordnet if self._wordnet else None

    @staticmethod
    def _get_synonyms(word: str, wordnet) -> List[str]:
        """Return a list of synonyms for *word* (excluding the word itself)."""
        synonyms: List[str] = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    synonyms.append(name)
        return list(set(synonyms))


# ── Paraphrase templates ───────────────────────────────────────────────────────

_PARAPHRASE_PATTERNS = [
    (r"^What (is|are) (.+)\?$", "Tell me about {1}."),
    (r"^How (do|can) I (.+)\?$", "What is the process to {1}?"),
    (r"^Can I (.+)\?$", "Is it possible to {0}?"),
    (r"^Tell me about (.+)$", "What should I know about {0}?"),
    (r"^(.+) side effects$", "What are the side effects of {0}?"),
]


def paraphrase(text: str) -> Optional[str]:
    """
    Apply a simple regex-based paraphrase rule to *text*.

    Returns a paraphrased string or ``None`` if no rule matches.
    """
    for pattern, template in _PARAPHRASE_PATTERNS:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            result = template
            for i, group in enumerate(groups):
                result = result.replace("{" + str(i) + "}", group)
            return result
    return None
