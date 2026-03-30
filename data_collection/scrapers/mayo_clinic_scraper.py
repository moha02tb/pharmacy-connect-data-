"""
Scraper for Mayo Clinic drug, supplement, and health-condition content.
"""

import logging
from typing import Dict, List

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class MayoClinicScraper(BaseScraper):
    """
    Scrapes drug information, side-effects, dosage, and drug-interaction
    content from the Mayo Clinic website.

    Falls back to an empty list when network access or optional dependencies
    are unavailable.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        logger.info("MayoClinicScraper initialised (source: %s)", self.base_url)

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
        """Extract structured drug/health content from a parsed Mayo page."""
        records: List[Dict] = []

        # Mayo Clinic wraps drug content in <div class="content">
        content_div = (
            soup.find("div", {"class": "content"})
            or soup.find("div", {"id": "main-content"})
            or soup.find("main")
            or soup
        )

        title_tag = soup.find("h1")
        page_title = title_tag.get_text().strip() if title_tag else "Mayo Clinic Article"

        # Each drug section is typically wrapped in its own <section>
        sections = content_div.find_all("section")
        if sections:
            for section in sections:
                heading = section.find(["h2", "h3"])
                section_title = (
                    f"{page_title} – {heading.get_text().strip()}"
                    if heading
                    else page_title
                )
                paragraphs = [
                    p.get_text().strip()
                    for p in section.find_all("p")
                    if len(p.get_text().strip()) > 20
                ]
                lists = []
                for ul in section.find_all("ul"):
                    lists.extend(
                        "• " + li.get_text().strip()
                        for li in ul.find_all("li")
                        if li.get_text().strip()
                    )
                content = " ".join(paragraphs + lists)
                if len(content) >= 50:
                    records.append(
                        self._build_record(
                            url=url,
                            title=section_title,
                            content=content,
                            category=self._classify_category(section_title, content),
                        )
                    )
        else:
            # Fallback: whole page
            paragraphs = [
                p.get_text().strip()
                for p in content_div.find_all("p")
                if len(p.get_text().strip()) > 30
            ]
            content = " ".join(paragraphs)
            if content:
                records.append(
                    self._build_record(
                        url=url,
                        title=page_title,
                        content=content,
                        category=self._classify_category(page_title, content),
                    )
                )

        return records

    @staticmethod
    def _classify_category(title: str, content: str) -> str:
        """Heuristically classify content into a category."""
        text = (title + " " + content).lower()
        if any(kw in text for kw in ("side effect", "adverse", "reaction")):
            return "side_effects"
        if any(kw in text for kw in ("dosage", "dose", "how to take", "instructions")):
            return "dosage"
        if any(kw in text for kw in ("interaction", "drug interaction")):
            return "drug_interaction"
        if any(kw in text for kw in ("supplement", "vitamin", "mineral", "herb")):
            return "medication"
        if any(kw in text for kw in ("drug", "medication", "medicine", "prescription")):
            return "medication"
        return "general_health"
