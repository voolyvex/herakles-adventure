"""Tests for the pure ranking metrics.

Expected values are hand-computed and written out longhand, so a test failure
points at the implementation rather than at a second implementation of the same
formula. Nothing here loads a model, a vector store, or touches the network.
"""

from __future__ import annotations

import math

import pytest

from myth_eval.metrics import (
    dcg,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    unanswerable_precision,
)


class TestDCG:
    def test_dcg_of_a_single_grade_two_document_is_three(self):
        """(2**2 - 1) / log2(2) == 3 / 1 == 3."""
        assert dcg([2]) == pytest.approx(3.0)

    def test_dcg_of_a_single_grade_one_document_is_one(self):
        """(2**1 - 1) / log2(2) == 1 / 1 == 1."""
        assert dcg([1]) == pytest.approx(1.0)

    def test_dcg_of_a_grade_zero_document_is_zero(self):
        """(2**0 - 1) == 0, so an irrelevant document contributes nothing."""
        assert dcg([0]) == pytest.approx(0.0)

    def test_dcg_discounts_by_log2_of_one_based_rank_plus_one(self):
        """[2, 1] == 3/log2(2) + 1/log2(3) == 3 + 0.6309297535714575."""
        expected = 3.0 / math.log2(2) + 1.0 / math.log2(3)

        assert dcg([2, 1]) == pytest.approx(expected)
        assert dcg([2, 1]) == pytest.approx(3.6309297535714578)

    def test_dcg_of_an_empty_ranking_is_zero(self):
        assert dcg([]) == pytest.approx(0.0)

    def test_grade_two_is_worth_more_than_three_times_grade_one(self):
        """Exponential gain: "directly answers" dominates "related context"."""
        assert dcg([2]) == pytest.approx(3.0 * dcg([1]))


class TestNDCGHandComputed:
    """nDCG against fully hand-worked values."""

    GRADES = {"d1": 2, "d2": 0, "d3": 1, "d4": 2}
    RANKED = ["d1", "d2", "d3"]

    def test_ndcg_at_3_matches_hand_computation(self):
        """actual = 3/log2(2) + 0/log2(3) + 1/log2(4) = 3 + 0 + 0.5 = 3.5
        ideal  = grades sorted desc [2, 2, 1]
               = 3/log2(2) + 3/log2(3) + 1/log2(4)
               = 3 + 1.8927892607143717 + 0.5 = 5.392789260714372
        nDCG   = 3.5 / 5.392789260714372 = 0.6490147918587163
        """
        actual = 3.0 + 0.0 + 0.5
        ideal = 3.0 + 3.0 / math.log2(3) + 0.5

        assert actual == pytest.approx(3.5)
        assert ideal == pytest.approx(5.392789260714372)
        assert ndcg_at_k(self.RANKED, self.GRADES, 3) == pytest.approx(
            0.6490147918587163
        )

    def test_ndcg_at_5_equals_ndcg_at_3_when_only_three_results_exist(self):
        """K beyond the result list adds nothing to actual, and the ideal is
        already exhausted at 3 judged-relevant documents."""
        assert ndcg_at_k(self.RANKED, self.GRADES, 5) == pytest.approx(
            0.6490147918587163
        )

    def test_ndcg_at_10_equals_ndcg_at_3_for_the_same_reason(self):
        assert ndcg_at_k(self.RANKED, self.GRADES, 10) == pytest.approx(
            0.6490147918587163
        )


class TestNDCGBounds:
    def test_the_ideal_ranking_scores_exactly_one(self):
        grades = {"a": 2, "b": 2, "c": 1}

        assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)

    def test_a_reversed_ranking_scores_materially_lower(self):
        """[b(1), a(2)] vs ideal [a(2), b(1)]:
        actual = 1/log2(2) + 3/log2(3) = 1 + 1.8927892607143717
        ideal  = 3/log2(2) + 1/log2(3) = 3 + 0.6309297535714575
        nDCG   = 2.8927892607143717 / 3.6309297535714578 = 0.7967075809905066
        """
        grades = {"a": 2, "b": 1}
        reversed_score = ndcg_at_k(["b", "a"], grades, 2)

        assert reversed_score == pytest.approx(0.7967075809905066)
        assert reversed_score < ndcg_at_k(["a", "b"], grades, 2)

    def test_ndcg_never_exceeds_one_even_with_repeated_documents(self):
        """Chunks share source documents, so an arm can return one twice.
        Counting it twice would push DCG past the ideal."""
        grades = {"a": 2, "b": 1}

        assert ndcg_at_k(["a", "a", "a"], grades, 3) <= 1.0

    def test_rank_one_beats_rank_nine_for_the_same_document(self):
        """The reason nDCG is the headline metric rather than Recall."""
        grades = {"target": 2}
        filler = [f"pad{i}" for i in range(8)]

        early = ndcg_at_k(["target"] + filler, grades, 10)
        late = ndcg_at_k(filler + ["target"], grades, 10)

        assert early == pytest.approx(1.0)
        assert late < early


class TestNDCGReadsAllThreeGrades:
    def test_grade_two_outranks_grade_one_which_outranks_grade_zero(self):
        grades = {"two": 2, "one": 1, "zero": 0}

        best = ndcg_at_k(["two", "one", "zero"], grades, 3)
        middling = ndcg_at_k(["one", "two", "zero"], grades, 3)
        worst = ndcg_at_k(["zero", "one", "two"], grades, 3)

        assert best > middling > worst

    def test_grade_one_is_distinguished_from_grade_zero(self):
        """The whole point of a graded scale over a binary one."""
        grades = {"related": 1, "irrelevant": 0, "elsewhere": 2}

        assert ndcg_at_k(["related"], grades, 1) > ndcg_at_k(["irrelevant"], grades, 1)


class TestNDCGEdgeCases:
    def test_an_empty_result_list_scores_zero(self):
        assert ndcg_at_k([], {"a": 2}, 5) == pytest.approx(0.0)

    def test_a_result_list_where_every_grade_is_zero_scores_zero(self):
        grades = {"a": 0, "b": 0, "relevant_elsewhere": 2}

        assert ndcg_at_k(["a", "b"], grades, 5) == pytest.approx(0.0)

    def test_unjudged_documents_score_zero_rather_than_erroring(self):
        """Pooled judging's defined behaviour on unseen results."""
        grades = {"judged": 2}

        assert ndcg_at_k(["never_seen"], grades, 5) == pytest.approx(0.0)

    def test_a_mix_of_judged_and_unjudged_counts_only_the_judged(self):
        grades = {"judged": 2}

        assert ndcg_at_k(["unjudged", "judged"], grades, 5) == pytest.approx(
            ndcg_at_k(["ignored_name", "judged"], grades, 5)
        )

    def test_ties_in_grades_are_handled(self):
        """Two equally-graded documents: either order scores the same."""
        grades = {"a": 2, "b": 2}

        assert ndcg_at_k(["a", "b"], grades, 2) == pytest.approx(
            ndcg_at_k(["b", "a"], grades, 2)
        )
        assert ndcg_at_k(["a", "b"], grades, 2) == pytest.approx(1.0)

    def test_k_larger_than_the_number_of_results_returned(self):
        grades = {"a": 2, "b": 1}

        assert ndcg_at_k(["a"], grades, 100) == pytest.approx(
            ndcg_at_k(["a"], grades, 2)
        )

    def test_k_larger_than_the_number_of_relevant_documents(self):
        """Ideal truncates at the number of relevant documents, so a perfect
        short ranking still scores 1.0."""
        grades = {"a": 2}

        assert ndcg_at_k(["a", "pad1", "pad2"], grades, 10) == pytest.approx(1.0)

    def test_no_relevant_document_anywhere_scores_zero(self):
        assert ndcg_at_k(["a"], {}, 5) == pytest.approx(0.0)

    def test_k_of_zero_scores_zero(self):
        assert ndcg_at_k(["a"], {"a": 2}, 0) == pytest.approx(0.0)


class TestRecall:
    def test_finding_both_relevant_documents_is_full_recall(self):
        grades = {"a": 2, "b": 1}

        assert recall_at_k(["a", "b"], grades, 5) == pytest.approx(1.0)

    def test_finding_one_of_two_relevant_documents_is_half_recall(self):
        grades = {"a": 2, "b": 1}

        assert recall_at_k(["a", "irrelevant"], grades, 5) == pytest.approx(0.5)

    def test_recall_at_k_respects_the_cutoff(self):
        """b sits at rank 3, so it is outside k=2."""
        grades = {"a": 2, "b": 1}

        assert recall_at_k(["a", "pad", "b"], grades, 2) == pytest.approx(0.5)
        assert recall_at_k(["a", "pad", "b"], grades, 3) == pytest.approx(1.0)

    def test_recall_is_binary_and_ignores_the_grade_distinction(self):
        """A grade-1 and a grade-2 document count identically — which is why
        nDCG, not Recall, is the headline metric."""
        assert recall_at_k(["x"], {"x": 1}, 5) == recall_at_k(["y"], {"y": 2}, 5)

    def test_recall_of_an_empty_result_list_is_zero(self):
        assert recall_at_k([], {"a": 2}, 5) == pytest.approx(0.0)

    def test_recall_with_no_relevant_documents_is_zero(self):
        """Undefined denominator, reported as 0.0."""
        assert recall_at_k(["a"], {"a": 0}, 5) == pytest.approx(0.0)

    def test_recall_at_3_5_and_10_are_all_computable(self):
        grades = {f"d{i}": 1 for i in range(10)}
        ranked = [f"d{i}" for i in range(10)]

        assert recall_at_k(ranked, grades, 3) == pytest.approx(0.3)
        assert recall_at_k(ranked, grades, 5) == pytest.approx(0.5)
        assert recall_at_k(ranked, grades, 10) == pytest.approx(1.0)

    def test_repeated_documents_do_not_inflate_recall(self):
        grades = {"a": 2, "b": 2}

        assert recall_at_k(["a", "a", "a"], grades, 3) == pytest.approx(0.5)


class TestMRR:
    def test_a_relevant_document_at_rank_one_scores_one(self):
        assert mrr_at_k(["a"], {"a": 2}, 10) == pytest.approx(1.0)

    def test_a_relevant_document_at_rank_two_scores_one_half(self):
        assert mrr_at_k(["pad", "a"], {"a": 2}, 10) == pytest.approx(0.5)

    def test_a_relevant_document_at_rank_four_scores_one_quarter(self):
        ranked = ["p1", "p2", "p3", "a"]

        assert mrr_at_k(ranked, {"a": 1}, 10) == pytest.approx(0.25)

    def test_only_the_first_relevant_document_counts(self):
        grades = {"a": 2, "b": 2}

        assert mrr_at_k(["a", "b"], grades, 10) == pytest.approx(1.0)

    def test_no_relevant_document_within_k_scores_zero(self):
        """Padding must be distinct: repeats collapse, which would pull the
        relevant document back inside the cutoff."""
        ranked = [f"pad{i}" for i in range(10)] + ["a"]

        assert mrr_at_k(ranked, {"a": 2}, 10) == pytest.approx(0.0)

    def test_repeated_documents_collapse_before_the_cutoff_is_applied(self):
        """A document cannot be relevant twice, so repeats do not consume
        rank positions."""
        ranked = ["pad", "pad", "pad", "a"]

        assert mrr_at_k(ranked, {"a": 2}, 10) == pytest.approx(0.5)

    def test_mrr_of_an_empty_result_list_is_zero(self):
        assert mrr_at_k([], {"a": 2}, 10) == pytest.approx(0.0)

    def test_mrr_is_binary_and_ignores_the_grade_distinction(self):
        assert mrr_at_k(["x"], {"x": 1}, 10) == mrr_at_k(["y"], {"y": 2}, 10)

    def test_unjudged_documents_do_not_count_as_relevant(self):
        assert mrr_at_k(["unjudged", "a"], {"a": 2}, 10) == pytest.approx(0.5)


class TestUnanswerablePrecision:
    """The unanswerable stratum: returning nothing is the correct behaviour."""

    def test_returning_nothing_is_correct(self):
        assert unanswerable_precision([]) == pytest.approx(1.0)

    def test_returning_something_confidently_is_incorrect(self):
        assert unanswerable_precision(["a"], [0.9], confidence_threshold=0.5) == (
            pytest.approx(0.0)
        )

    def test_returning_only_low_confidence_results_counts_as_abstaining(self):
        result = unanswerable_precision(
            ["a", "b"], [0.2, 0.1], confidence_threshold=0.5
        )

        assert result == pytest.approx(1.0)

    def test_one_confident_result_among_several_is_enough_to_fail(self):
        result = unanswerable_precision(
            ["a", "b"], [0.1, 0.8], confidence_threshold=0.5
        )

        assert result == pytest.approx(0.0)

    def test_without_scores_any_returned_result_counts_as_an_answer(self):
        assert unanswerable_precision(["a"]) == pytest.approx(0.0)

    def test_a_result_exactly_at_the_threshold_is_not_confident(self):
        """Threshold is exclusive: scoring at the bar is not above it."""
        result = unanswerable_precision(["a"], [0.5], confidence_threshold=0.5)

        assert result == pytest.approx(1.0)


class TestPercentile:
    def test_p50_of_a_symmetric_sample_is_the_middle_value(self):
        assert percentile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)

    def test_p95_approaches_the_maximum(self):
        values = [float(i) for i in range(1, 101)]

        assert percentile(values, 0.95) == pytest.approx(95.05)

    def test_percentile_of_a_single_observation_is_that_observation(self):
        assert percentile([4.2], 0.95) == pytest.approx(4.2)

    def test_percentile_of_an_empty_sample_is_zero(self):
        assert percentile([], 0.5) == pytest.approx(0.0)

    def test_percentile_does_not_require_sorted_input(self):
        assert percentile([3.0, 1.0, 2.0], 0.5) == pytest.approx(2.0)


class TestNoHeavyDependencies:
    def test_the_metrics_module_imports_no_models_or_stores(self):
        """Seam 1 must run in a minimal environment: no torch, no chromadb,
        no sentence_transformers, not even numpy."""
        import myth_eval.metrics as metrics_module

        source = open(metrics_module.__file__).read()

        for forbidden in ("torch", "chromadb", "sentence_transformers", "numpy"):
            assert f"import {forbidden}" not in source
