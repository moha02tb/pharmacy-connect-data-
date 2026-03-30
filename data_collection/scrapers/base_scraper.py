"""
Base scraper class shared by all data source scrapers.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import re

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Abstract base class for all health-data scrapers."""

    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get("name", "Unknown Source")
        self.base_url = config.get("base_url", "")
        self.rate_limit = config.get("rate_limit_seconds", 2.0)
        self.timeout = config.get("timeout_seconds", 30)
        self._session = None

    # ── Public API ────────────────────────────────────────────────────────────

    def scrape(self) -> List[Dict]:
        """
        Run the scraper and return a list of scraped records.

        Each record is a dict with at minimum:
            - source (str)
            - url (str)
            - title (str)
            - content (str)
            - topics (list[str])
        """
        if not self.config.get("enabled", True):
            logger.info("Scraper %s is disabled – skipping.", self.name)
            return []

        logger.info("Starting scraper: %s", self.name)
        records: List[Dict] = []

        for endpoint in self.config.get("endpoints", []):
            url = self.base_url + endpoint
            try:
                page_records = self._scrape_endpoint(url)
                records.extend(page_records)
                logger.info(
                    "Scraped %d records from %s", len(page_records), url
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Error scraping %s: %s", url, exc)
            finally:
                time.sleep(self.rate_limit)

        logger.info(
            "Finished scraper %s – total records: %d", self.name, len(records)
        )
        return records

    # ── Subclass interface ────────────────────────────────────────────────────

    @abstractmethod
    def _scrape_endpoint(self, url: str) -> List[Dict]:
        """Scrape a single endpoint URL and return records."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_record(
        self,
        url: str,
        title: str,
        content: str,
        topics: Optional[List[str]] = None,
        category: str = "general_health",
    ) -> Dict:
        """Build a standardised data record dict."""
        return {
            "source": self.name,
            "url": url,
            "title": title.strip(),
            "content": self._clean_text(content),
            "topics": topics or self.config.get("topics", []),
            "category": category,
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove excessive whitespace and normalise line endings."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()
