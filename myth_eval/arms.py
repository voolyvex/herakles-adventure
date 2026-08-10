"""The configuration arms the harness compares.

Four arms at this stage: dense-only, sparse-only, hybrid, and
hybrid-plus-reranker. The fifth ``god_context`` arm lands separately.

Every arm implements the retrieval protocol, so the runner calls them
identically and metric code never branches on which one produced a result.

A note on the hybrid arm. Fusion merges the dense and sparse pools and sorts
the result by each hit's raw ``score`` — mixing bounded cosine similarity
(0 to 1) with unbounded BM25 scores, so BM25 dominates the ordering for reasons
that have nothing to do with relevance. **This is left unfixed deliberately.**
The first baseline captures the system as it currently behaves, defect
included, so that the follow-up fix can be demonstrated as a measured
improvement rather than asserted. Fixing it first would forfeit the clearest
available demonstration that the harness catches real defects. See the README.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from myth_eval.retrieval import RetrievedItem, Retriever

logger = logging.getLogger(__name__)

__all__ = [
    "HybridArm",
    "RerankedArm",
    "build_arms",
    "ARM_NAMES",
    "DENSE",
    "SPARSE",
]

# The canonical reporting order, and the single source of every arm name. The
# four below are unpacked from it rather than written out again, so no arm name
# is spelled twice in the codebase. The unpacking is positional: reordering the
# tuple reorders these bindings with it, so keep the two in step.
ARM_NAMES = ("dense", "sparse", "hybrid", "hybrid_rerank")
DENSE, SPARSE, HYBRID, HYBRID_RERANK = ARM_NAMES

# Candidate fan-out per sub-arm before fusion. Mirrors the orchestrator's
# hard-coded 40. Parameterising the orchestrator's own pools is out of scope
# for the MVP — only needed for pool-size sweeps later.
FANOUT = 40
RERANK_POOL = 30


def _deduplicate(items: Sequence[RetrievedItem]) -> List[RetrievedItem]:
    """Collapse hits sharing text, keeping the first. Mirrors the orchestrator."""
    seen = set()
    unique: List[RetrievedItem] = []
    for item in items:
        key = item.text or item.chunk_id
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _renumber(items: Sequence[RetrievedItem]) -> List[RetrievedItem]:
    """Reassign dense zero-based ranks after a merge or re-sort."""
    return [
        RetrievedItem(
            chunk_id=item.chunk_id,
            source_document=item.source_document,
            text=item.text,
            score=item.score,
            rank=rank,
            metadata=item.metadata,
        )
        for rank, item in enumerate(items)
    ]


class HybridArm:
    """Dense and sparse merged, then sorted by raw score.

    Reproduces the orchestrator's fusion behaviour, *including* the score-scale
    defect described in this module's docstring. That fidelity is the point:
    the arm must measure what production does, not an idealised version of it.
    """

    def __init__(
        self,
        dense: Retriever,
        sparse: Optional[Retriever],
        name: str = HYBRID,
        fanout: int = FANOUT,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self.name = name
        self._fanout = fanout

    def _candidates(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievedItem]:
        dense_hits = self._dense.retrieve(query, self._fanout, filters)
        sparse_hits = (
            self._sparse.retrieve(query, self._fanout, filters) if self._sparse else []
        )
        return _deduplicate(list(dense_hits) + list(sparse_hits))

    def fused_candidates(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        """The merged dense and sparse pool, score-ordered and untruncated.

        This is the whole of the fusion rule, and the only place it lives. The
        reranked arm needs the pool deeper than ``retrieve`` returns it and
        without ranks reassigned, so depth is the caller's business: slice what
        you need. Ranks are assigned by whoever returns items to a caller.

        Args:
            query: The natural-language query.
            filters: Optional metadata filter, passed to both sub-arms.

        Returns:
            Every deduplicated candidate, ordered by raw score, best first.
        """
        candidates = self._candidates(query, filters)
        # The defective sort, preserved on purpose: cosine similarity and BM25
        # are not on a common scale.
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        return _renumber(self.fused_candidates(query, filters)[:k])


class RerankedArm:
    """A hybrid pool re-ordered by the cross-encoder reranker.

    The reranker scores every candidate against the original query, which puts
    the pool back on one scale — so this arm does not inherit the fusion
    defect's ordering, though it does inherit which candidates reach it.
    """

    def __init__(
        self,
        hybrid: HybridArm,
        reranker: Any,
        name: str = HYBRID_RERANK,
        pool_size: int = RERANK_POOL,
    ) -> None:
        self._hybrid = hybrid
        self._reranker = reranker
        self.name = name
        self._pool_size = pool_size

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        pool = self._hybrid.fused_candidates(query, filters)[: self._pool_size]
        if not pool:
            return []

        # The reranker agent speaks dicts, so translate at this boundary and
        # translate back, keeping RetrievedItem the only type the runner sees.
        as_dicts = [
            {"text": item.text, "metadata": item.metadata, "id": item.chunk_id}
            for item in pool
        ]
        try:
            reranked = self._reranker.rerank(query, as_dicts, top_k=k)
        except Exception as error:  # pragma: no cover - defensive, mirrors orchestrator
            logger.warning("Reranking failed (%s); falling back to fused order", error)
            return _renumber(pool[:k])

        by_chunk_id = {item.chunk_id: item for item in pool}
        ordered: List[RetrievedItem] = []
        for entry in reranked:
            original = by_chunk_id.get(entry.get("id"))
            if original is None:
                continue
            ordered.append(
                RetrievedItem(
                    chunk_id=original.chunk_id,
                    source_document=original.source_document,
                    text=original.text,
                    score=float(entry.get("rerank_score", original.score)),
                    rank=0,
                    metadata=original.metadata,
                )
            )
        return _renumber(ordered[:k])


def build_arms(
    dense: Retriever,
    sparse: Optional[Retriever],
    reranker: Optional[Any] = None,
) -> List[Retriever]:
    """Assemble the configuration matrix from already-adapted retrievers.

    Args:
        dense: The dense arm, already behind the protocol.
        sparse: The sparse arm, or None if unavailable.
        reranker: A reranker agent exposing ``rerank(query, candidates, top_k)``.

    Returns:
        The arms, in a stable reporting order.
    """
    hybrid = HybridArm(dense, sparse)
    arms: List[Retriever] = [dense]
    if sparse is not None:
        arms.append(sparse)
    arms.append(hybrid)
    if reranker is not None:
        arms.append(RerankedArm(hybrid, reranker))
    return arms
