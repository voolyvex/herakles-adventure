"""Tests for the evaluation runner, driven entirely by a fake retriever.

Seam 2. Nothing here constructs ChromaDB, loads an embedding model, loads a
reranker, or touches the network — that is the property that makes the runner
testable at all, and one test asserts it directly.
"""

from __future__ import annotations

import json

import pytest

from myth_eval.arms import ARM_NAMES, HybridArm, RerankedArm, build_arms
from myth_eval.dataset import Dataset, Question, Stratum
from myth_eval.fakes import FakeRetriever
from myth_eval.retrieval import adapt_stack
from myth_eval.runner import (
    GATE_K,
    K_VALUES,
    EvaluationResults,
    evaluate_arm,
    run_evaluation,
)


@pytest.fixture
def dataset() -> Dataset:
    """A small labelled set covering every stratum."""
    return Dataset(
        questions=[
            Question(
                id="f1",
                question="who chased Daphne",
                stratum=Stratum.FACTUAL,
                relevance={"apollo.md": 2, "daphne.md": 1},
            ),
            Question(
                id="s1",
                question="how do transformation myths resolve",
                stratum=Stratum.SYNTHESIS,
                relevance={"apollo.md": 1, "midas.md": 2},
            ),
            Question(
                id="e1",
                question="who was Phaeton",
                stratum=Stratum.ENTITY,
                relevance={"phaeton.md": 2},
            ),
            Question(
                id="u1",
                question="who forged Mjolnir",
                stratum=Stratum.UNANSWERABLE,
            ),
        ],
        chunk_size_chars=1200,
        chunk_overlap_chars=200,
    )


@pytest.fixture
def perfect_arm() -> FakeRetriever:
    """Returns the best document first for every answerable question, and
    correctly abstains on the unanswerable one."""
    return FakeRetriever(
        "perfect",
        responses={
            "who chased Daphne": ["apollo.md", "daphne.md"],
            "how do transformation myths resolve": ["midas.md", "apollo.md"],
            "who was Phaeton": ["phaeton.md"],
            "who forged Mjolnir": [],
        },
    )


@pytest.fixture
def weak_arm() -> FakeRetriever:
    """Finds relevant documents late, and answers the unanswerable question."""
    return FakeRetriever(
        "weak",
        responses={
            "who chased Daphne": ["noise1.md", "noise2.md", "apollo.md"],
            "how do transformation myths resolve": ["noise1.md", "midas.md"],
            "who was Phaeton": ["noise1.md"],
            "who forged Mjolnir": ["noise1.md"],
        },
    )


class TestTheRunnerVisitsEveryArm:
    def test_every_configured_arm_produces_a_result(self, dataset, perfect_arm, weak_arm):
        results = run_evaluation([perfect_arm, weak_arm], dataset)

        assert [arm.name for arm in results.arms] == ["perfect", "weak"]

    def test_every_arm_is_asked_every_question(self, dataset, perfect_arm, weak_arm):
        run_evaluation([perfect_arm, weak_arm], dataset)

        for arm in (perfect_arm, weak_arm):
            asked = [call["query"] for call in arm.calls]
            assert asked == [q.question for q in dataset]

    def test_the_four_arm_matrix_runs(self, dataset):
        """dense, sparse, hybrid, hybrid_rerank — the arms this ticket covers."""
        arms = [FakeRetriever(name) for name in ARM_NAMES]

        results = run_evaluation(arms, dataset)

        assert len(results.arms) == 4
        assert [a.name for a in results.arms] == list(ARM_NAMES)


class TestReportedMetrics:
    def test_each_arm_reports_ndcg_recall_at_3_5_and_10_plus_mrr_at_10(
        self, dataset, perfect_arm
    ):
        result = run_evaluation([perfect_arm], dataset).arms[0]

        for k in K_VALUES:
            assert f"ndcg@{k}" in result.metrics
            assert f"recall@{k}" in result.metrics
        assert "mrr@10" in result.metrics

    def test_each_arm_reports_p50_and_p95_latency(self, dataset, perfect_arm):
        result = run_evaluation([perfect_arm], dataset).arms[0]

        assert "p50" in result.latency
        assert "p95" in result.latency
        assert result.latency["p95"] >= 0.0

    def test_a_better_arm_scores_higher_on_the_gate_metric(
        self, dataset, perfect_arm, weak_arm
    ):
        results = run_evaluation([perfect_arm, weak_arm], dataset)
        scores = results.aggregate_gate_scores()

        assert scores["perfect"] > scores["weak"]

    def test_the_gate_metric_is_aggregate_ndcg_at_5(self, dataset, perfect_arm):
        results = run_evaluation([perfect_arm], dataset)

        assert results.gate_metric == f"ndcg@{GATE_K}"
        assert GATE_K == 5

    def test_the_winning_configuration_is_identified(self, dataset, perfect_arm, weak_arm):
        results = run_evaluation([weak_arm, perfect_arm], dataset)

        assert results.best_arm() == "perfect"

    def test_a_perfect_arm_scores_one_on_ndcg(self, dataset, perfect_arm):
        result = run_evaluation([perfect_arm], dataset).arms[0]

        assert result.metrics["ndcg@5"] == pytest.approx(1.0)


class TestUnjudgedResultsScoreZero:
    def test_an_arm_returning_only_unjudged_documents_scores_zero(self, dataset):
        arm = FakeRetriever(
            "unjudged",
            responses={q.question: ["never_labelled.md"] for q in dataset},
        )

        result = run_evaluation([arm], dataset).arms[0]

        assert result.metrics["ndcg@5"] == pytest.approx(0.0)
        assert result.metrics["recall@5"] == pytest.approx(0.0)

    def test_unjudged_documents_do_not_raise(self, dataset):
        arm = FakeRetriever(
            "unjudged",
            responses={q.question: ["a.md", "b.md", "c.md"] for q in dataset},
        )

        run_evaluation([arm], dataset)  # must not raise


class TestStrataAreIndicativeDiagnostics:
    def test_per_stratum_numbers_are_reported(self, dataset, perfect_arm):
        result = run_evaluation([perfect_arm], dataset).arms[0]

        assert Stratum.FACTUAL in result.strata
        assert Stratum.SYNTHESIS in result.strata
        assert Stratum.ENTITY in result.strata

    def test_strata_are_explicitly_marked_indicative_in_the_output(
        self, dataset, perfect_arm
    ):
        payload = run_evaluation([perfect_arm], dataset).to_dict()

        note = payload["arms"][0]["strata"]["_note"]

        assert "indicative" in note.lower()
        assert "never a gate" in note.lower()

    def test_the_gate_reads_the_aggregate_not_a_stratum(self, dataset, perfect_arm):
        """A noisy stratum must not be able to block a merge on its own."""
        results = run_evaluation([perfect_arm], dataset)

        assert results.to_dict()["gate_metric"] == "aggregate ndcg@5"


class TestUnanswerableStratum:
    def test_abstaining_scores_full_marks(self, dataset, perfect_arm):
        result = run_evaluation([perfect_arm], dataset).arms[0]

        assert result.strata[Stratum.UNANSWERABLE][
            "unanswerable_precision"
        ] == pytest.approx(1.0)

    def test_confidently_answering_scores_zero(self, dataset, weak_arm):
        result = run_evaluation([weak_arm], dataset).arms[0]

        assert result.strata[Stratum.UNANSWERABLE][
            "unanswerable_precision"
        ] == pytest.approx(0.0)

    def test_the_unanswerable_stratum_does_not_drag_down_ndcg(self, dataset, perfect_arm):
        """nDCG is undefined for a question with no relevant document, so
        averaging a zero in from that stratum would understate every arm."""
        result = run_evaluation([perfect_arm], dataset).arms[0]

        assert result.metrics["ndcg@5"] == pytest.approx(1.0)


class TestMachineReadableOutput:
    def test_results_are_written_as_valid_json(self, dataset, perfect_arm, tmp_path):
        results = run_evaluation([perfect_arm], dataset)
        path = results.save(tmp_path / "results.json")

        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["arms"][0]["arm"] == "perfect"
        assert payload["question_count"] == 4

    def test_saved_results_round_trip(self, dataset, perfect_arm, tmp_path):
        path = run_evaluation([perfect_arm], dataset).save(tmp_path / "r.json")

        assert EvaluationResults.load(path)["best_arm"] == "perfect"

    def test_output_records_chunker_provenance(self, dataset, perfect_arm, tmp_path):
        payload = run_evaluation([perfect_arm], dataset).to_dict()

        assert payload["provenance"]["chunk_size_chars"] == 1200
        assert payload["provenance"]["chunk_overlap_chars"] == 200

    def test_output_is_stable_across_runs_for_diffing(self, dataset, tmp_path):
        """Results must be diffable, so key order cannot wander."""
        arm = FakeRetriever("a", responses={q.question: ["apollo.md"] for q in dataset})

        first = json.dumps(run_evaluation([arm], dataset).to_dict(), sort_keys=False)
        second = json.dumps(run_evaluation([arm], dataset).to_dict(), sort_keys=False)

        # Latency differs run to run; compare everything else.
        strip = lambda blob: json.loads(blob)  # noqa: E731
        a, b = strip(first), strip(second)
        for payload in (a, b):
            for entry in payload["arms"]:
                entry.pop("latency_seconds")

        assert a == b


class TestUnlabelledDatasetIsFlagged:
    def test_an_unlabelled_dataset_is_reported_as_such(self, perfect_arm):
        """Before the grading pass every metric reads zero. The output must say
        so, or a reader will mistake an ungraded run for a catastrophic one."""
        unlabelled = Dataset(
            questions=[
                Question(id="f1", question="who chased Daphne", stratum=Stratum.FACTUAL)
            ]
        )

        results = run_evaluation([perfect_arm], unlabelled)

        assert results.dataset_labelled is False
        assert results.to_dict()["dataset_labelled"] is False

    def test_a_labelled_dataset_is_reported_as_labelled(self, dataset, perfect_arm):
        assert run_evaluation([perfect_arm], dataset).dataset_labelled is True


class TestHybridArmFusion:
    def test_hybrid_merges_both_sub_arms(self, dataset):
        dense = FakeRetriever("dense", responses={"who was Phaeton": ["phaeton.md"]})
        sparse = FakeRetriever("sparse", responses={"who was Phaeton": ["midas.md"]})

        results = HybridArm(dense, sparse).retrieve("who was Phaeton", k=10)

        assert {item.source_document for item in results} == {
            "phaeton.md",
            "midas.md",
        }

    def test_hybrid_sorts_by_raw_score_preserving_the_known_defect(self):
        """The fusion defect is deliberately baselined, not fixed: a raw BM25
        score outranks a bounded cosine score regardless of relevance."""
        dense = FakeRetriever(
            "dense", responses={"q": ["relevant.md"]}, scores={"q": [0.95]}
        )
        sparse = FakeRetriever(
            "sparse", responses={"q": ["bm25_noise.md"]}, scores={"q": [14.2]}
        )

        results = HybridArm(dense, sparse).retrieve("q", k=10)

        assert results[0].source_document == "bm25_noise.md"
        assert results[0].rank == 0

    def test_hybrid_works_when_sparse_is_unavailable(self, dataset):
        dense = FakeRetriever("dense", responses={"who was Phaeton": ["phaeton.md"]})

        results = HybridArm(dense, None).retrieve("who was Phaeton", k=5)

        assert [item.source_document for item in results] == ["phaeton.md"]

    def test_hybrid_deduplicates_documents_found_by_both_arms(self):
        dense = FakeRetriever("dense", responses={"q": ["same.md"]})
        sparse = FakeRetriever("sparse", responses={"q": ["same.md"]})

        results = HybridArm(dense, sparse).retrieve("q", k=10)

        assert len(results) == 1

    def test_hybrid_reassigns_dense_ranks_after_fusion(self):
        dense = FakeRetriever("dense", responses={"q": ["a.md", "b.md"]})
        sparse = FakeRetriever("sparse", responses={"q": ["c.md"]})

        results = HybridArm(dense, sparse).retrieve("q", k=10)

        assert [item.rank for item in results] == list(range(len(results)))


class TestTheFusedCandidatePool:
    """Fusion through the hybrid arm's own interface, with no reranker in sight.

    The reranked arm needs this pool unsorted-by-rank and deeper than what a
    caller of ``retrieve`` gets back. It used to reach into a private attribute
    and re-sort, which put the fusion rule — and the score-scale defect it
    carries — in two places at once.
    """

    def test_the_pool_is_ordered_by_raw_score(self):
        """The preserved defect, asserted directly: BM25 outranks cosine.

        14.2 beats 0.95 because the two are not on a common scale, not because
        the BM25 hit is more relevant. See this module's docstring in arms.py.
        """
        dense = FakeRetriever(
            "dense", responses={"q": ["cosine.md"]}, scores={"q": [0.95]}
        )
        sparse = FakeRetriever(
            "sparse", responses={"q": ["bm25_noise.md"]}, scores={"q": [14.2]}
        )

        pool = HybridArm(dense, sparse).fused_candidates("q")

        assert [item.source_document for item in pool] == [
            "bm25_noise.md",
            "cosine.md",
        ]

    def test_the_pool_is_not_truncated_to_a_retrieval_depth(self):
        """Deeper than ``retrieve`` returns — that is why it exists."""
        dense = FakeRetriever(
            "dense", responses={"q": [f"d{i:02d}.md" for i in range(12)]}
        )
        arm = HybridArm(dense, None)

        assert len(arm.fused_candidates("q")) == 12
        assert len(arm.retrieve("q", k=5)) == 5

    def test_the_pool_deduplicates_across_both_sub_arms(self):
        dense = FakeRetriever("dense", responses={"q": ["same.md"]})
        sparse = FakeRetriever("sparse", responses={"q": ["same.md"]})

        assert len(HybridArm(dense, sparse).fused_candidates("q")) == 1

    def test_retrieve_returns_the_head_of_the_same_pool(self):
        """One fusion rule: ``retrieve`` is the pool, sliced and renumbered."""
        dense = FakeRetriever(
            "dense", responses={"q": ["a.md", "b.md"]}, scores={"q": [0.9, 0.8]}
        )
        sparse = FakeRetriever("sparse", responses={"q": ["c.md"]}, scores={"q": [7.0]})
        arm = HybridArm(dense, sparse)

        pool = arm.fused_candidates("q")
        retrieved = arm.retrieve("q", k=2)

        assert [item.source_document for item in retrieved] == [
            item.source_document for item in pool[:2]
        ]
        assert [item.rank for item in retrieved] == [0, 1]

    def test_the_pool_survives_sparse_being_unavailable(self):
        dense = FakeRetriever("dense", responses={"q": ["a.md"]})

        pool = HybridArm(dense, None).fused_candidates("q")

        assert [item.source_document for item in pool] == ["a.md"]


class TestRerankedArm:
    def test_it_takes_its_pool_through_the_hybrid_arms_interface(self):
        """No private attribute, and no second sort of its own."""

        class RecordingHybrid(HybridArm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pool_requests = 0

            def fused_candidates(self, query, filters=None):
                self.pool_requests += 1
                return super().fused_candidates(query, filters)

        class FakeReranker:
            def rerank(self, query, candidates, top_k=5):
                return [
                    {**c, "rerank_score": 1.0 - i}
                    for i, c in enumerate(candidates[:top_k])
                ]

        dense = FakeRetriever("dense", responses={"q": ["a.md"]})
        hybrid = RecordingHybrid(dense, None)

        RerankedArm(hybrid, FakeReranker()).retrieve("q", k=5)

        assert hybrid.pool_requests == 1

    def test_the_reranker_reorders_the_fused_pool(self):
        class FakeReranker:
            def rerank(self, query, candidates, top_k=5):
                # Reverse the pool, to prove the arm honours the reranker.
                reversed_pool = list(reversed(candidates))[:top_k]
                return [
                    {**c, "rerank_score": 1.0 - i} for i, c in enumerate(reversed_pool)
                ]

        dense = FakeRetriever("dense", responses={"q": ["a.md"]}, scores={"q": [0.9]})
        sparse = FakeRetriever("sparse", responses={"q": ["b.md"]}, scores={"q": [12.0]})
        arm = RerankedArm(HybridArm(dense, sparse), FakeReranker())

        results = arm.retrieve("q", k=5)

        # Fusion puts b.md first (raw BM25); the reranker reverses it.
        assert [item.source_document for item in results] == ["a.md", "b.md"]

    def test_a_failing_reranker_falls_back_to_the_fused_order(self):
        class BrokenReranker:
            def rerank(self, query, candidates, top_k=5):
                raise RuntimeError("model unavailable")

        dense = FakeRetriever("dense", responses={"q": ["a.md"]})
        arm = RerankedArm(HybridArm(dense, None), BrokenReranker())

        assert [item.source_document for item in arm.retrieve("q", k=5)] == ["a.md"]

    def test_an_empty_pool_reranks_to_nothing(self):
        class FakeReranker:
            def rerank(self, query, candidates, top_k=5):
                raise AssertionError("should not be called for an empty pool")

        arm = RerankedArm(HybridArm(FakeRetriever("dense"), None), FakeReranker())

        assert arm.retrieve("q", k=5) == []


class TestBuildArms:
    def test_the_matrix_is_assembled_in_a_stable_order(self):
        class FakeReranker:
            def rerank(self, query, candidates, top_k=5):
                return []

        arms = build_arms(
            FakeRetriever("dense"), FakeRetriever("sparse"), FakeReranker()
        )

        assert [a.name for a in arms] == list(ARM_NAMES)

    def test_sparse_absence_drops_that_arm_rather_than_failing(self):
        arms = build_arms(FakeRetriever("dense"), None, None)

        assert [a.name for a in arms] == ["dense", "hybrid"]


class TestBothPathsShareOneAssembly:
    """The claim ``--fake`` makes: identical code path, assembly included."""

    def test_the_scripted_matrix_is_assembled_by_build_arms(self, dataset):
        """Same names, same types, same order as the live matrix would give.

        The scripted path used to hand-write these four arms, agreeing with
        ``build_arms`` only because the same literals appeared in both places.
        """
        from myth_eval.cli import build_fake_arms

        arms = build_fake_arms(dataset)

        assert [arm.name for arm in arms] == list(ARM_NAMES)
        assert isinstance(arms[2], HybridArm)
        assert isinstance(arms[3], RerankedArm)

    def test_the_scripted_matrix_still_retrieves(self, dataset):
        """Assembly changed; behaviour did not."""
        from myth_eval.cli import build_fake_arms

        results = run_evaluation(build_fake_arms(dataset), dataset)

        assert [arm.name for arm in results.arms] == list(ARM_NAMES)


class TestAdaptingARetrievalStack:
    """The adapter seam the live command sits on, exercised without the stack.

    ``adapt_stack`` takes anything exposing ``dense``, ``sparse`` and
    ``reranker``, which is all the command ever needed from ``RAGSystem``. That
    is what puts the refusal below within reach of a test that loads no models.
    """

    class Stack:
        def __init__(self, dense=None, sparse=None, reranker=None):
            self.dense = dense
            self.sparse = sparse
            self.reranker = reranker

    def test_it_refuses_when_the_sparse_retriever_failed_to_initialise(self):
        """The harness's loudest safety behaviour, finally executed.

        Silently dropping sparse here would score the sparse and hybrid arms
        zero for reasons unrelated to retrieval quality, and that number would
        land in the committed baseline. Refusing is the whole point.
        """
        stack = self.Stack(dense=object(), sparse=None, reranker=object())

        with pytest.raises(RuntimeError) as raised:
            adapt_stack(stack)

        assert "sparse" in str(raised.value).lower()

    def test_the_refusal_reports_nothing_rather_than_zeros(self):
        """No partial matrix escapes: the caller gets an exception, not arms."""
        stack = self.Stack(dense=object(), sparse=None, reranker=object())

        try:
            adapted = adapt_stack(stack)
        except RuntimeError:
            adapted = None

        assert adapted is None

    def test_a_whole_stack_adapts_to_the_protocol(self):
        reranker = object()
        stack = self.Stack(dense=object(), sparse=object(), reranker=reranker)

        dense, sparse, adapted_reranker = adapt_stack(stack)

        assert (dense.name, sparse.name) == ("dense", "sparse")
        assert adapted_reranker is reranker

    def test_dropping_the_reranker_leaves_the_other_two_adapted(self):
        stack = self.Stack(dense=object(), sparse=object(), reranker=object())

        dense, sparse, reranker = adapt_stack(stack, with_reranker=False)

        assert reranker is None
        assert (dense.name, sparse.name) == ("dense", "sparse")


class TestNoHeavyDependencies:
    def test_the_runner_test_suite_loads_no_models_or_stores(self):
        """The property that makes Seam 2 useful."""
        import sys

        for forbidden in ("chromadb", "sentence_transformers", "flashrank"):
            assert forbidden not in sys.modules, (
                f"{forbidden} was imported; the runner suite must stay free of it"
            )
