"""A fake retriever implementing the retrieval protocol.

This is what makes the runner testable at all. With it, the whole harness —
configuration matrix, pooling, aggregation, per-stratum breakdown, baseline
comparison — runs with no ChromaDB, no embedding model, no reranker and no
network, in milliseconds.

It lives in the package rather than in the test file because the candidate-pool
generator and the baseline-comparison tests use it too, and because it
documents, by example, exactly what a new backend has to implement. It is the
same seam a future Azure retriever slots into.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from myth_eval.retrieval import RetrievedItem

__all__ = ["FakeRetriever"]


class FakeRetriever:
    """Returns pre-scripted results per question.

    Args:
        name: The arm name reported in results.
        responses: Map of question id to the source documents this arm returns,
            best first. A question with no entry returns nothing.
        scores: Optional map of question id to per-result scores, aligned with
            ``responses``. Defaults to a descending ramp, which is enough for
            the unanswerable stratum's confidence check.
        record_calls: Whether to retain the arguments of every call, so tests
            can assert the runner visited each arm with the expected inputs.
    """

    def __init__(
        self,
        name: str,
        responses: Optional[Mapping[str, Sequence[str]]] = None,
        scores: Optional[Mapping[str, Sequence[float]]] = None,
        record_calls: bool = True,
    ) -> None:
        self.name = name
        self._responses: Dict[str, List[str]] = {
            key: list(value) for key, value in (responses or {}).items()
        }
        self._scores: Dict[str, List[float]] = {
            key: list(value) for key, value in (scores or {}).items()
        }
        self.record_calls = record_calls
        self.calls: List[Dict[str, Any]] = []

    def set_response(
        self,
        question_id: str,
        documents: Sequence[str],
        scores: Optional[Sequence[float]] = None,
    ) -> None:
        """Script this arm's answer for one question."""
        self._responses[question_id] = list(documents)
        if scores is not None:
            self._scores[question_id] = list(scores)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        """Return the scripted results for ``query``.

        The query text is used as the lookup key, so tests script responses
        against the question text itself. Question ids also work, since scripts
        commonly key on whichever is more readable.
        """
        if self.record_calls:
            self.calls.append({"query": query, "k": k, "filters": filters})

        documents = self._responses.get(query, [])
        scores = self._scores.get(query)

        items: List[RetrievedItem] = []
        for rank, document in enumerate(documents[:k]):
            if scores is not None and rank < len(scores):
                score = float(scores[rank])
            else:
                # Descending ramp, so rank order and score order agree.
                score = 1.0 - (rank * 0.1)
            items.append(
                RetrievedItem(
                    chunk_id=f"{document}_c{rank}",
                    source_document=document,
                    text=f"fake passage {rank} from {document}",
                    score=score,
                    rank=rank,
                    metadata={"source_file": document, "god": "unknown"},
                )
            )
        return items
