"""
Data collection scrapers / loaders package.

Provides loaders for the MedDialog and DrugBank data sources.
The legacy web scrapers (Red Cross, CDC, Mayo Clinic) have been removed
because their endpoints were unreliable (403 / 404 / rate-limiting).
``base_scraper.py`` is kept as a template for adding new scraping-based
sources in the future.
"""

from .meddialog_loader import MedDialogLoader
from .meddialog_kb_builder import MedDialogKBBuilder
from .drugbank_loader import DrugBankLoader

__all__ = ["MedDialogLoader", "MedDialogKBBuilder", "DrugBankLoader"]
