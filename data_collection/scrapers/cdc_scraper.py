"""
Scraper for CDC (Centers for Disease Control and Prevention) health content.
"""

import logging
from typing import Dict, List

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class CDCScraper(BaseScraper):
    """
    Scrapes medication safety, vaccination, drug-overdose prevention, and
    public-health content from the CDC website.

    Falls back to an empty list when network access or optional dependencies
    are unavailable.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        logger.info("CDCScraper initialised (source: %s)", self.base_url)

    # ── BaseScraper interface ─────────────────────────────────────────────────

    def _scrape_endpoint(self, url: str) -> List[Dict]:
        """Attempt to scrape ``url`` and return structured data records."""
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
        """Extract health-content blocks from a parsed CDC page."""
        records: List[Dict] = []

        # CDC wraps content in <div class="syndicate"> or <main>
        main_content = soup.find("main") or soup.find(
            "div", {"class": "syndicate"}
        )
        container = main_content or soup

        # Extract sections separated by <h2>/<h3> headings
        headings = container.find_all(["h2", "h3"])
        if headings:
            for heading in headings:
                title = heading.get_text().strip()
                content_parts: List[str] = []
                sibling = heading.find_next_sibling()
                while sibling and sibling.name not in ("h2", "h3"):
                    if sibling.name == "p":
                        text = sibling.get_text().strip()
                        if text:
                            content_parts.append(text)
                    elif sibling.name == "ul":
                        for li in sibling.find_all("li"):
                            item = li.get_text().strip()
                            if item:
                                content_parts.append("• " + item)
                    sibling = sibling.find_next_sibling()

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
            # Fallback: grab all paragraphs
            paragraphs = [
                p.get_text().strip()
                for p in container.find_all("p")
                if len(p.get_text().strip()) > 40
            ]
            if paragraphs:
                title_tag = soup.find("h1")
                title = title_tag.get_text().strip() if title_tag else "CDC Health Information"
                records.append(
                    self._build_record(
                        url=url,
                        title=title,
                        content=" ".join(paragraphs),
                        category=self._classify_category(title, " ".join(paragraphs)),
                    )
                )

        return records

    @staticmethod
    def _classify_category(title: str, content: str) -> str:
        """Heuristically classify content into a category."""
        text = (title + " " + content).lower()
        if any(kw in text for kw in ("vaccine", "vaccination", "immunization")):
            return "vaccination"
        if any(kw in text for kw in ("overdose", "opioid", "naloxone")):
            return "medication"
        if any(kw in text for kw in ("prescription", "drug", "medication")):
            return "medication"
        if any(kw in text for kw in ("interaction", "side effect", "adverse")):
            return "drug_interaction"
        return "general_health"
