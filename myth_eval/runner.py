"""The evaluation runner: one command over the whole configuration matrix.

Runs every configured arm across the question set, scores each with the pure
metric functions, and writes results machine-readably so they can be diffed,
archived and compared by tooling.

The runner is driven entirely through the :class:`~myth_eval.retrieval.Retriever`
protocol, so it can be exercised end to end against a fake retriever — no
ChromaDB, no embedding model, no reranker, no network. That is the second test
seam, and it is the same seam a future Azure backend slots into.

Per-stratum numbers are computed and reported but are marked **indicative**:
with eight to fifteen questions per stratum, a single question flipping moves a
stratum by several points. They are diagnostics, never gates. The gate is
aggregate nDCG@5.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from myth_eval.dataset import Dataset, Question, Stratum
from myth_eval.metrics import (
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    unanswerable_precision,
)
from myth_eval.pool import POOL_DEPTH, ArmRetrievals
from myth_eval.retrieval import RetrievedItem, Retriever

logger = logging.getLogger(__name__)

__all__ = [
    "K_VALUES",
    "GATE_K",
    "QuestionOutcome",
    "ArmResult",
    "EvaluationResults",
    "evaluate_arm",
    "run_evaluation",
    "default_results_path",
]

# 3 is what production requests, 10 is the pooling depth, and 5 sits between
# them as the gating figure.
K_VALUES = (3, 5, 10)
GATE_K = 5

# Per-stratum figures are reported with this marker attached, so a reader cannot
# mistake a diagnostic for a gate.
INDICATIVE_NOTE = (
    "indicative only: 8-15 questions per stratum means a single question "
    "flipping moves this by several points; never a gate"
)


def default_results_path() -> Path:
    return Path(__file__).resolve().parent.parent / "eval_data" / "results.json"


@dataclass
class QuestionOutcome:
    """One arm's scored result for one question.

    Scoring reads ``documents`` and ``scores``; ``retrieved_for_pooling`` is
    named for what it is — a handover, not part of the score. Pooling is its
    only reader, and it holds whole passages rather than filenames because a
    human grading a candidate needs the text, which a filename cannot supply.
    Carrying it through the run is what lets a pool be built from one
    evaluation pass instead of a second round of retrieval.

    Held in memory only: ``ArmResult.to_dict`` does not serialise outcomes, so
    the results file stays a compact metrics artefact and the handover never
    reaches it.
    """

    question_id: str
    stratum: str
    documents: List[str]
    scores: List[float]
    latency_seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)
    retrieved_for_pooling: List[RetrievedItem] = field(default_factory=list)


@dataclass
class ArmResult:
    """One configuration arm's scores across the whole question set."""

    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)
    strata: Dict[str, Dict[str, float]] = field(default_factory=dict)
    outcomes: List[QuestionOutcome] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "arm": self.name,
            "metrics": {k: round(v, 6) for k, v in sorted(self.metrics.items())},
            "latency_seconds": {
                k: round(v, 6) for k, v in sorted(self.latency.items())
            },
            "strata": {
                "_note": INDICATIVE_NOTE,
                **{
                    stratum: {k: round(v, 6) for k, v in sorted(scores.items())}
                    for stratum, scores in sorted(self.strata.items())
                },
            },
        }
        if self.diagnostics:
            data["diagnostics"] = self.diagnostics
        return data


@dataclass
class EvaluationResults:
    """The full matrix: every arm over the question set."""

    arms: List[ArmResult] = field(default_factory=list)
    question_count: int = 0
    dataset_labelled: bool = True
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def gate_metric(self) -> str:
        return f"ndcg@{GATE_K}"

    def aggregate_gate_scores(self) -> Dict[str, float]:
        """Each arm's gating figure — aggregate nDCG@5."""
        return {arm.name: arm.metrics.get(self.gate_metric, 0.0) for arm in self.arms}

    def best_arm(self) -> Optional[str]:
        """Which configuration wins on the gating metric."""
        scores = self.aggregate_gate_scores()
        return max(scores, key=scores.get) if scores else None

    def retrievals_for_pooling(self) -> List[ArmRetrievals]:
        """This run's retrieved passages, in the shape pooling takes.

        The handover ``QuestionOutcome.retrieved_for_pooling`` exists for.
        Building a pool from this reads the pass that has already happened,
        rather than paying for a second round of retrieval.
        """
        return [
            ArmRetrievals(
                name=arm.name,
                retrievals={
                    o.question_id: list(o.retrieved_for_pooling)
                    for o in arm.outcomes
                },
            )
            for arm in self.arms
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_metric": f"aggregate {self.gate_metric}",
            "question_count": self.question_count,
            "dataset_labelled": self.dataset_labelled,
            "provenance": self.provenance,
            "best_arm": self.best_arm(),
            "arms": [arm.to_dict() for arm in self.arms],
        }

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path is not None else default_results_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path) -> Dict[str, Any]:
        """Load a results file as a plain dict, for baseline comparison."""
        return json.loads(Path(path).read_text(encoding="utf-8"))


def _score_question(
    question: Question,
    documents: Sequence[str],
    scores: Sequence[float],
) -> Dict[str, float]:
    """Score one question's returned documents against its ground truth.

    Unanswerable questions are scored differently by necessity: there is no
    relevant document, so recall has an empty denominator and nDCG has no ideal
    ranking to normalise against. The measure is whether the system abstained.
    """
    if question.is_unanswerable:
        return {
            "unanswerable_precision": unanswerable_precision(documents, scores),
        }

    metrics: Dict[str, float] = {}
    for k in K_VALUES:
        metrics[f"ndcg@{k}"] = ndcg_at_k(documents, question.relevance, k)
        metrics[f"recall@{k}"] = recall_at_k(documents, question.relevance, k)
    metrics["mrr@10"] = mrr_at_k(documents, question.relevance, 10)
    return metrics


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(outcomes: Sequence[QuestionOutcome]) -> Dict[str, float]:
    """Mean each metric over the outcomes that report it.

    Metrics are averaged only over questions that produce them, so the
    unanswerable stratum does not drag down nDCG by contributing zeros for a
    metric that is undefined for it.
    """
    keys = {key for outcome in outcomes for key in outcome.metrics}
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [o.metrics[key] for o in outcomes if key in o.metrics]
        aggregated[key] = _mean(values)
    return aggregated


def evaluate_arm(
    retriever: Retriever,
    dataset: Dataset,
    k: int = POOL_DEPTH,
    filters_for: Optional[Callable[[Question], Optional[Dict[str, Any]]]] = None,
) -> ArmResult:
    """Run one arm across the question set and score it.

    Args:
        retriever: Anything implementing the retrieval protocol.
        dataset: The question set.
        k: Retrieval depth. Defaults to :data:`~myth_eval.pool.POOL_DEPTH`, so
            that metrics at every K value and a full-depth pool are all
            computable from one pass.
        filters_for: Optional per-question filter, used by the god_context arm.

    Returns:
        The arm's aggregate metrics, latency percentiles, per-stratum
        breakdowns and per-question outcomes.
    """
    outcomes: List[QuestionOutcome] = []
    latencies: List[float] = []

    for question in dataset:
        filters = filters_for(question) if filters_for else None

        started = time.perf_counter()
        items: List[RetrievedItem] = retriever.retrieve(question.question, k, filters)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        documents = [item.source_document for item in items]
        scores = [item.score for item in items]

        outcomes.append(
            QuestionOutcome(
                question_id=question.id,
                stratum=question.stratum,
                documents=documents,
                scores=scores,
                latency_seconds=elapsed,
                metrics=_score_question(question, documents, scores),
                retrieved_for_pooling=list(items),
            )
        )

    strata: Dict[str, Dict[str, float]] = {}
    for stratum in Stratum.ALL:
        in_stratum = [o for o in outcomes if o.stratum == stratum]
        if in_stratum:
            strata[stratum] = _aggregate(in_stratum)

    return ArmResult(
        name=getattr(retriever, "name", "unnamed"),
        metrics=_aggregate(outcomes),
        latency={
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "mean": _mean(latencies),
        },
        strata=strata,
        outcomes=outcomes,
    )


def run_evaluation(
    retrievers: Sequence[Retriever],
    dataset: Dataset,
    k: int = POOL_DEPTH,
    filters_for: Optional[Callable[[Question], Optional[Dict[str, Any]]]] = None,
) -> EvaluationResults:
    """Run every arm over the question set.

    Args:
        retrievers: The configuration matrix.
        dataset: The question set.
        k: Retrieval depth per arm.
        filters_for: Optional per-question filter applied to every arm. Arms
            needing their own filtering carry it internally instead.

    Returns:
        The full results.
    """
    arms = [evaluate_arm(r, dataset, k, filters_for) for r in retrievers]

    if not dataset.is_labelled:
        logger.warning(
            "Dataset has no relevance labels; every ranking metric will read "
            "zero. Run the grading pass before treating these as scores."
        )

    return EvaluationResults(
        arms=arms,
        question_count=len(dataset),
        dataset_labelled=dataset.is_labelled,
        provenance={
            "chunk_size_chars": dataset.chunk_size_chars,
            "chunk_overlap_chars": dataset.chunk_overlap_chars,
            "retrieval_depth": k,
            "k_values": list(K_VALUES),
        },
    )
