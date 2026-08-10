"""The single command that runs the whole configuration matrix.

    python -m myth_eval.cli --output eval_data/results.json

Adding ``--pool-output`` also emits the candidate pool for manual grading,
built from the union of every arm's top-10 results. It reads the run that has
already happened, so pooling costs no extra retrieval:

    python -m myth_eval.cli --pool-output

Constructing the real retrieval stack is deferred into :func:`build_live_arms`,
which is the only place in the harness that imports torch, chromadb or the
agents. Everything above it — runner, metrics, dataset, arms — is driven through
the retrieval protocol, so ``--fake`` runs the identical code path against
scripted retrievers in milliseconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from myth_eval.dataset import Dataset, default_dataset_path
from myth_eval.pool import build_pool, default_pool_path
from myth_eval.runner import (
    POOL_DEPTH,
    EvaluationResults,
    default_results_path,
    run_evaluation,
)

logger = logging.getLogger(__name__)

__all__ = ["build_live_arms", "main"]


def build_live_arms(
    embedding_model: Optional[str] = None,
    with_reranker: bool = True,
) -> List[Any]:
    """Construct the real four-arm matrix against the built index.

    Imports the heavy stack lazily, so importing this module stays cheap.
    """
    from myth_eval.arms import build_arms
    from myth_eval.index import DEFAULT_EMBEDDING_MODEL, index_root
    from myth_eval.nltk_resources import ensure_nltk_resources
    from myth_eval.retrieval import DenseRetrieverAdapter, SparseRetrieverAdapter

    # Fail loudly here rather than silently losing the sparse arm: RAGSystem
    # swallows a tokeniser failure into a warning, which would score sparse and
    # hybrid zero for reasons unrelated to retrieval quality.
    ensure_nltk_resources()

    # Resolve the store once and both open and report that same path, so the
    # log cannot name a directory other than the one actually read.
    store = index_root()

    from rag_system import RAGSystem

    system = RAGSystem(
        embedding_model_name=embedding_model or DEFAULT_EMBEDDING_MODEL,
        store_dir=str(store),
    )

    logger.info("Index in use: %s", store)

    dense = DenseRetrieverAdapter(system.agentic_rag.dense)
    sparse_agent = system.agentic_rag.sparse
    if sparse_agent is None:
        raise RuntimeError(
            "The sparse retriever failed to initialise, so the sparse and "
            "hybrid arms cannot be measured. Refusing to produce results that "
            "would silently score them zero."
        )
    sparse = SparseRetrieverAdapter(sparse_agent)

    reranker = system.agentic_rag.reranker if with_reranker else None
    return build_arms(dense, sparse, reranker)


def build_fake_arms(dataset: Dataset) -> List[Any]:
    """A scripted four-arm matrix, for exercising the command with no index."""
    from myth_eval.arms import HybridArm, RerankedArm
    from myth_eval.fakes import FakeRetriever

    responses = {}
    for question in dataset:
        if question.is_unanswerable:
            responses[question.question] = []
        else:
            responses[question.question] = sorted(question.relevance)

    dense = FakeRetriever("dense", responses=responses)
    sparse = FakeRetriever("sparse", responses=responses)
    hybrid = HybridArm(dense, sparse)

    class PassthroughReranker:
        def rerank(self, query, candidates, top_k=5):
            return [
                {**c, "rerank_score": 1.0 - i} for i, c in enumerate(candidates[:top_k])
            ]

    return [dense, sparse, hybrid, RerankedArm(hybrid, PassthroughReranker())]


def _print_summary(results: EvaluationResults) -> None:
    """Human-readable summary. The JSON file is the machine-readable artefact."""
    gate = results.gate_metric
    print(f"\nQuestions: {results.question_count}   Gate: aggregate {gate}")
    if not results.dataset_labelled:
        print(
            "WARNING: dataset carries no relevance labels — every ranking "
            "metric below reads zero. Run the grading pass first."
        )

    header = f"{'arm':<16}{gate:>10}{'recall@5':>10}{'mrr@10':>10}{'p50 s':>9}{'p95 s':>9}"
    print(header)
    print("-" * len(header))
    for arm in results.arms:
        print(
            f"{arm.name:<16}"
            f"{arm.metrics.get(gate, 0.0):>10.4f}"
            f"{arm.metrics.get('recall@5', 0.0):>10.4f}"
            f"{arm.metrics.get('mrr@10', 0.0):>10.4f}"
            f"{arm.latency.get('p50', 0.0):>9.3f}"
            f"{arm.latency.get('p95', 0.0):>9.3f}"
        )

    best = results.best_arm()
    if best:
        print(f"\nBest on {gate}: {best}")
    print("\nPer-stratum figures are in the results file and are indicative only.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the retrieval evaluation matrix over the question set.",
    )
    parser.add_argument("--dataset", default=None, help="Question set JSON.")
    parser.add_argument("--output", default=None, help="Where to write results JSON.")
    parser.add_argument("--k", type=int, default=POOL_DEPTH, help="Retrieval depth.")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip the reranked arm (faster; useful when iterating).",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Run against scripted retrievers instead of the real index.",
    )
    parser.add_argument(
        "--pool-output",
        default=None,
        help=(
            "Also write the candidate pool for manual grading to this path "
            "(default: eval_data/candidate_pool.json). Pooling reuses this "
            "run's results, so it costs no extra retrieval."
        ),
        nargs="?",
        const="",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset_path = Path(args.dataset) if args.dataset else default_dataset_path()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    dataset = Dataset.load(dataset_path)

    arms = (
        build_fake_arms(dataset)
        if args.fake
        else build_live_arms(args.embedding_model, not args.no_reranker)
    )

    results = run_evaluation(arms, dataset, k=args.k)
    output = Path(args.output) if args.output else default_results_path()
    written = results.save(output)

    _print_summary(results)
    print(f"\nResults written to {written}")

    if args.pool_output is not None:
        # The spec fixes the pooling depth at 10. A deeper retrieval run must
        # not widen the pool past it, so --k can only ever pool shallower.
        pool = build_pool(results.arms, dataset, depth=min(args.k, POOL_DEPTH))
        pool_path = Path(args.pool_output) if args.pool_output else default_pool_path()
        pool_written = pool.save(pool_path)
        print(
            f"Candidate pool: {pool.candidate_count()} grading decisions over "
            f"{len(pool.gradeable)} questions "
            f"({len(pool.unique_documents)} distinct documents) "
            f"-> {pool_written}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
