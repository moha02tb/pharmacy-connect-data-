"""
Scraper for American Red Cross health and first-aid content.
"""

import logging
from typing import Dict, List

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class RedCrossScraper(BaseScraper):
    """
    Scrapes first-aid, medication safety, and emergency-preparedness content
    from the American Red Cross website.

    In production this class makes real HTTP requests; during testing or when
    the network is unavailable, ``_scrape_endpoint`` falls back gracefully to
    returning an empty list so the rest of the pipeline can still run.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        logger.info("RedCrossScraper initialised (source: %s)", self.base_url)

    # ── BaseScraper interface ─────────────────────────────────────────────────

    def _scrape_endpoint(self, url: str) -> List[Dict]:
        """
        Attempt to scrape ``url`` and return a list of data records.

        Requires the optional ``requests`` and ``beautifulsoup4`` packages.
        Falls back to an empty list when they are not installed or the network
        is unavailable.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning(
                "requests/beautifulsoup4 not installed – skipping %s", url
            )
            return []

        try:
            headers = {"User-Agent": "PharmacyConnectBot/1.0"}
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch %s: %s", url, exc)
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        return self._parse_page(url, soup)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_page(self, url: str, soup) -> List[Dict]:
        """Extract articles / content blocks from a parsed Red Cross page."""
        records: List[Dict] = []

        # Red Cross pages use <article> tags and heading+paragraph combos.
        articles = soup.find_all("article")
        if articles:
            for article in articles:
                title_tag = article.find(["h1", "h2", "h3"])
                title = title_tag.get_text() if title_tag else "Red Cross Article"
                content_parts = [
                    p.get_text() for p in article.find_all("p") if p.get_text().strip()
                ]
                content = " ".join(content_parts)
                if len(content) >= 50:
                    records.append(
                        self._build_record(
                            url=url,
                            title=title,
                            content=content,
                            category=self._classify_category(title, content),
                        )
                    )
        else:
            # Fallback: grab main content paragraphs
            paragraphs = soup.find_all("p")
            content = " ".join(
                p.get_text() for p in paragraphs if len(p.get_text().strip()) > 40
            )
            title_tag = soup.find("h1")
            title = title_tag.get_text() if title_tag else "Red Cross Content"
            if content:
                records.append(
                    self._build_record(
                        url=url,
                        title=title,
                        content=content,
                        category=self._classify_category(title, content),
                    )
                )

        return records

    @staticmethod
    def _classify_category(title: str, content: str) -> str:
        """Heuristically classify a piece of content into a category."""
        text = (title + " " + content).lower()
        if any(kw in text for kw in ("medication", "drug", "medicine", "prescription")):
            return "medication"
        if any(kw in text for kw in ("first aid", "bleeding", "wound", "burn")):
            return "first_aid"
        if any(kw in text for kw in ("emergency", "disaster", "preparedness")):
            return "emergency"
        if any(kw in text for kw in ("cpr", "aed", "cardiac")):
            return "first_aid"
        return "general_health"
