"""Tests for the evaluation runner, driven entirely by a fake retriever.

Seam 2. Nothing here constructs ChromaDB, loads an embedding model, loads a
reranker, or touches the network — that is the property that makes the runner
testable at all, and one test asserts it directly.
"""

from __future__ import annotations

import json

import pytest

from myth_eval.arms import HybridArm, RerankedArm, build_arms
from myth_eval.dataset import Dataset, Question, Stratum
from myth_eval.fakes import FakeRetriever
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
        arms = [FakeRetriever(name) for name in ("dense", "sparse", "hybrid", "hybrid_rerank")]

        results = run_evaluation(arms, dataset)

        assert len(results.arms) == 4
        assert [a.name for a in results.arms] == [
            "dense",
            "sparse",
            "hybrid",
            "hybrid_rerank",
        ]


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


class TestRerankedArm:
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

        assert [a.name for a in arms] == ["dense", "sparse", "hybrid", "hybrid_rerank"]

    def test_sparse_absence_drops_that_arm_rather_than_failing(self):
        arms = build_arms(FakeRetriever("dense"), None, None)

        assert [a.name for a in arms] == ["dense", "hybrid"]


class TestNoHeavyDependencies:
    def test_the_runner_test_suite_loads_no_models_or_stores(self):
        """The property that makes Seam 2 useful."""
        import sys

        for forbidden in ("chromadb", "sentence_transformers", "flashrank"):
            assert forbidden not in sys.modules, (
                f"{forbidden} was imported; the runner suite must stay free of it"
            )
