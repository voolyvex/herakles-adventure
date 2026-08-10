"""The retrieval protocol and the adapters that normalise existing agents onto it.

The two retrievers this project already has are not interchangeable:

- ``DenseRetrieverAgent.retrieve`` takes a query *string* and returns tidy
  records: ``{id, text, metadata, score, query_used}``, with ``source_file``
  nested inside ``metadata``.
- ``SparseRetrieverAgent.retrieve`` takes a *pre-tokenised term list* and
  returns the whole raw lore chunk splatted out, so ``source_file``, ``god``
  and ``title`` sit at the top level and ``metadata`` is a sibling key.

No metric can score both arms until those differences are erased. This module
defines one protocol (:class:`Retriever`) and one uniform result type
(:class:`RetrievedItem`), and adapts each existing agent onto them.

The adapters **wrap**; they never reach into the agents or modify them. The
working application keeps calling the agents directly and is unaffected.

``RetrievedItem.source_document`` is the identity that ground-truth relevance
labels attach to. Labels are recorded against source documents rather than
chunk IDs because chunk IDs are a function of the chunker's size and overlap
settings, so labelling against them means any re-chunk silently rots the
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

__all__ = [
    "RetrievedItem",
    "Retriever",
    "DenseRetrieverAdapter",
    "SparseRetrieverAdapter",
    "adapt_stack",
    "normalise_hit",
]


@dataclass(frozen=True)
class RetrievedItem:
    """One retrieved passage, in the single shape all metric code reads.

    Attributes:
        chunk_id: The chunk's identifier. Useful for debugging and for
            deduplicating within a single arm's results. Deliberately *not*
            what relevance labels key on.
        source_document: The corpus file the chunk came from. This is the
            identity relevance labels attach to, so labels survive re-chunking.
        text: The passage text.
        score: The retriever's own score. Not comparable across arms — dense
            returns a bounded cosine similarity and sparse an unbounded BM25
            score — so it is carried for diagnostics, never for cross-arm
            ranking.
        rank: Zero-based position in the arm's returned ordering. This is what
            ranking metrics consume, because it is comparable across arms
            whereas ``score`` is not.
        metadata: The chunk's remaining metadata, flattened into one mapping.
    """

    chunk_id: str
    source_document: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """The one shape evaluation code calls retrieval through.

    Implemented by the adapters below, by the hybrid arms, by the fake
    retriever used in tests, and — the reason this is a protocol rather than a
    base class — by a future Azure AI Search backend, which can then be scored
    on the same dataset without the harness changing.
    """

    name: str

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        """Return up to ``k`` passages for ``query``, best first.

        Args:
            query: The natural-language query. Implementations that need a
                different input form (the sparse arm needs a token list) do
                that conversion internally — callers always pass a string.
            k: Maximum number of results to return.
            filters: Optional metadata filter, e.g. ``{"god": {"$in": [...]}}``.

        Returns:
            At most ``k`` items ordered best-first, with ``rank`` assigned
            densely from zero.
        """
        ...


def _coalesce_metadata(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a hit's nested ``metadata`` and its top-level keys into one map.

    Dense hits carry their fields under ``metadata``; sparse hits splat the raw
    chunk at the top level. Merging both means downstream code reads one shape.
    Nested ``metadata`` wins on conflict, since that is the curated mapping.
    """
    merged: Dict[str, Any] = {
        key: value
        for key, value in hit.items()
        if key not in {"metadata", "text", "id", "score"}
    }
    nested = hit.get("metadata")
    if isinstance(nested, dict):
        merged.update(nested)
    return merged


def normalise_hit(hit: Dict[str, Any], rank: int) -> RetrievedItem:
    """Convert one raw agent hit into a :class:`RetrievedItem`.

    Handles both hit shapes, since ``source_file`` may live at the top level
    (sparse) or nested under ``metadata`` (dense).

    Args:
        hit: A raw result dict from either retriever agent.
        rank: Zero-based position in the arm's ordering.

    Returns:
        The normalised item.
    """
    metadata = _coalesce_metadata(hit)

    source_document = (
        hit.get("source_file")
        or (hit.get("metadata") or {}).get("source_file")
        or metadata.get("source_file")
        or ""
    )

    chunk_id = hit.get("id") or metadata.get("id") or ""

    # A missing score is treated as 0.0 rather than an error: the sparse arm can
    # legitimately score a document zero, and metrics rank on `rank` anyway.
    raw_score = hit.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0

    return RetrievedItem(
        chunk_id=str(chunk_id),
        source_document=str(source_document),
        text=hit.get("text", "") or "",
        score=score,
        rank=rank,
        metadata=metadata,
    )


def normalise_hits(hits: List[Dict[str, Any]], k: int) -> List[RetrievedItem]:
    """Normalise an ordered list of raw hits, truncating to ``k``."""
    return [normalise_hit(hit, rank) for rank, hit in enumerate(hits[:k])]


class DenseRetrieverAdapter:
    """Adapts :class:`DenseRetrieverAgent` onto the :class:`Retriever` protocol.

    The dense agent already accepts a query string, so this adapter is mostly
    about normalising the returned shape and pinning the arm's name.
    """

    def __init__(self, dense_agent: Any, name: str = "dense") -> None:
        """Wrap a dense retriever agent without modifying it.

        Args:
            dense_agent: An object exposing
                ``retrieve(query, k, where_filter) -> list[dict]``.
            name: The arm name reported in results.
        """
        self._agent = dense_agent
        self.name = name

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        hits = self._agent.retrieve(query, k, filters)
        return normalise_hits(hits or [], k)


class SparseRetrieverAdapter:
    """Adapts :class:`SparseRetrieverAgent` onto the :class:`Retriever` protocol.

    The tokenisation difference is absorbed here. The sparse agent takes a
    pre-tokenised term list, so this adapter tokenises the query string before
    delegating — callers pass a string to every arm, exactly as the protocol
    promises.

    Tokenisation mirrors the agent's own ``_tokenize`` (NLTK word tokens,
    lowercased) so that query terms match how the corpus was indexed. Term
    expansion with Greek/Roman name variants is deliberately *not* done here:
    that is the orchestrator's behaviour, and this adapter is the bare
    sparse-only arm.
    """

    def __init__(self, sparse_agent: Any, name: str = "sparse") -> None:
        """Wrap a sparse retriever agent without modifying it.

        Args:
            sparse_agent: An object exposing
                ``retrieve(query_terms, k, where_filter) -> list[dict]``.
            name: The arm name reported in results.
        """
        self._agent = sparse_agent
        self.name = name

    def _tokenize(self, query: str) -> List[str]:
        """Tokenise a query string the way the wrapped agent tokenised the corpus."""
        tokenize = getattr(self._agent, "_tokenize", None)
        if callable(tokenize):
            return tokenize(query)
        # Fall back to a whitespace split if the agent exposes no tokeniser.
        # Keeps the adapter usable against fakes in tests without requiring NLTK.
        return query.lower().split()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        hits = self._agent.retrieve(self._tokenize(query), k, filters)
        return normalise_hits(hits or [], k)


def adapt_stack(
    stack: Any,
    with_reranker: bool = True,
) -> Tuple[Retriever, Retriever, Optional[Any]]:
    """Put a retrieval stack's three agents behind the protocol.

    ``stack`` is anything exposing ``dense``, ``sparse`` and ``reranker`` —
    which is the whole of what the harness ever wanted from ``RAGSystem``.
    Taking the duck type rather than the concrete class is what keeps this
    function, and the refusal below, reachable without torch or chromadb.

    Args:
        stack: An object with ``dense``, ``sparse`` and ``reranker`` attributes.
        with_reranker: Whether to carry the reranker through. False drops the
            reranked arm, which is faster when iterating.

    Returns:
        The dense adapter, the sparse adapter, and the reranker or None.

    Raises:
        RuntimeError: If the stack's sparse retriever is absent. ``RAGSystem``
            swallows a tokeniser failure into a warning, so the sparse agent
            going missing is silent at its source; refusing here is what stops
            a run scoring the sparse and hybrid arms zero for reasons that have
            nothing to do with retrieval quality. Dense has no such failure
            mode — ``RAGSystem`` cannot come up without it — so its absence is
            a programming error and surfaces as ``AttributeError``. The
            reranker may legitimately be absent, because dropping the reranked
            arm is a supported choice.
    """
    sparse_agent = getattr(stack, "sparse", None)
    if sparse_agent is None:
        raise RuntimeError(
            "The sparse retriever failed to initialise, so the sparse and "
            "hybrid arms cannot be measured. Refusing to produce results that "
            "would silently score them zero."
        )

    dense = DenseRetrieverAdapter(stack.dense)
    sparse = SparseRetrieverAdapter(sparse_agent)
    reranker = getattr(stack, "reranker", None) if with_reranker else None
    return dense, sparse, reranker
