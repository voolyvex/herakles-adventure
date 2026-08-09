"""NLTK resource bootstrap for the evaluation harness.

The sparse arm tokenises with ``nltk.word_tokenize``. The existing agents guard
that with a ``nltk.data.find('tokenizers/punkt')`` check and download ``punkt``
if it is missing — but NLTK 3.8.2+ splits the tokeniser tables into a separate
``punkt_tab`` resource, which ``word_tokenize`` also requires. So the guard
passes, the download is skipped, and tokenisation still raises.

RAGSystem swallows that failure into a warning ("Failed to initialize
SparseRetrieverAgent... Continuing without sparse retrieval") and carries on
with no sparse arm at all. For the application that is a quiet degradation; for
this harness it would be a silent corruption — the sparse-only and hybrid arms
would score zero for a reason that has nothing to do with retrieval quality,
and the committed baseline would encode it.

Calling :func:`ensure_nltk_resources` before building retrievers makes the
requirement explicit rather than depending on ambient machine state.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

logger = logging.getLogger(__name__)

__all__ = ["REQUIRED_RESOURCES", "ensure_nltk_resources", "missing_resources"]

# (resource id, the path nltk.data.find uses to locate it).
# punkt_tab is the one the existing agents miss.
REQUIRED_RESOURCES: Sequence[tuple] = (
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
)


def missing_resources() -> List[str]:
    """Return the ids of required NLTK resources not present locally."""
    import nltk

    missing = []
    for resource_id, lookup_path in REQUIRED_RESOURCES:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            missing.append(resource_id)
    return missing


def ensure_nltk_resources(quiet: bool = True) -> List[str]:
    """Download any missing NLTK resources the sparse arm needs.

    Args:
        quiet: Passed through to ``nltk.download``.

    Returns:
        The ids that had to be downloaded. Empty when everything was present.

    Raises:
        RuntimeError: If tokenisation still fails afterwards. The harness fails
            loudly here rather than silently scoring the sparse arm zero.
    """
    import nltk

    downloaded = []
    for resource_id in missing_resources():
        logger.info("Downloading missing NLTK resource: %s", resource_id)
        nltk.download(resource_id, quiet=quiet)
        downloaded.append(resource_id)

    try:
        nltk.word_tokenize("Apollo pursued Daphne.")
    except LookupError as error:
        raise RuntimeError(
            "NLTK tokenisation is unavailable after attempting to download "
            f"{[r for r, _ in REQUIRED_RESOURCES]}. The sparse arm cannot run, "
            "and scoring it as zero would corrupt the baseline. Original "
            f"error: {error}"
        ) from error

    return downloaded
