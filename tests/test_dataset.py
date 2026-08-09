"""Tests for the dataset types: strata, grading, and provenance."""

from __future__ import annotations

import json

import pytest

from myth_eval.dataset import Dataset, Question, Stratum


class TestQuestion:
    def test_an_unanswerable_question_is_labelled_by_construction(self):
        """An empty relevance map *is* the ground truth for a question where
        returning nothing is correct — not an absence of one."""
        question = Question(id="u1", question="who forged Mjolnir", stratum=Stratum.UNANSWERABLE)

        assert question.is_unanswerable
        assert question.is_labelled

    def test_an_ungraded_answerable_question_is_not_labelled(self):
        question = Question(id="f1", question="who chased Daphne", stratum=Stratum.FACTUAL)

        assert not question.is_labelled

    def test_a_graded_answerable_question_is_labelled(self):
        question = Question(
            id="f1",
            question="who chased Daphne",
            stratum=Stratum.FACTUAL,
            relevance={"apollo.md": 2},
        )

        assert question.is_labelled

    def test_relevant_documents_respects_the_threshold(self):
        question = Question(
            id="f1",
            question="q",
            stratum=Stratum.FACTUAL,
            relevance={"direct.md": 2, "related.md": 1, "irrelevant.md": 0},
        )

        assert question.relevant_documents(threshold=1) == ["direct.md", "related.md"]
        assert question.relevant_documents(threshold=2) == ["direct.md"]

    def test_a_question_round_trips_through_a_dict(self):
        original = Question(
            id="f1",
            question="who chased Daphne",
            stratum=Stratum.FACTUAL,
            relevance={"apollo.md": 2},
            source_document="005_APOLLO_AND_DAPHNE.md",
            notes="uses vocabulary absent from the source passage",
        )

        assert Question.from_dict(original.to_dict()) == original


class TestDataset:
    @pytest.fixture
    def dataset(self) -> Dataset:
        return Dataset(
            questions=[
                Question(id="f1", question="a", stratum=Stratum.FACTUAL, relevance={"x.md": 2}),
                Question(id="f2", question="b", stratum=Stratum.FACTUAL, relevance={"y.md": 1}),
                Question(id="s1", question="c", stratum=Stratum.SYNTHESIS, relevance={"z.md": 2}),
                Question(id="u1", question="d", stratum=Stratum.UNANSWERABLE),
            ]
        )

    def test_questions_can_be_selected_by_stratum(self, dataset):
        assert [q.id for q in dataset.by_stratum(Stratum.FACTUAL)] == ["f1", "f2"]

    def test_stratum_counts_are_reported(self, dataset):
        assert dataset.stratum_counts() == {
            Stratum.FACTUAL: 2,
            Stratum.SYNTHESIS: 1,
            Stratum.UNANSWERABLE: 1,
        }

    def test_a_fully_graded_dataset_is_labelled(self, dataset):
        assert dataset.is_labelled

    def test_one_ungraded_question_makes_the_dataset_unlabelled(self, dataset):
        dataset.questions.append(Question(id="f3", question="e", stratum=Stratum.FACTUAL))

        assert not dataset.is_labelled

    def test_an_empty_dataset_is_not_labelled(self):
        assert not Dataset().is_labelled

    def test_chunker_parameters_are_recorded_as_provenance(self, dataset, tmp_path):
        """Labels attach to source documents so they survive re-chunking, but a
        reader still needs to know which regime the pool was gathered under."""
        dataset.chunk_size_chars = 900
        dataset.chunk_overlap_chars = 150

        path = dataset.save(tmp_path / "questions.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["provenance"]["chunk_size_chars"] == 900
        assert payload["provenance"]["chunk_overlap_chars"] == 150

    def test_a_dataset_round_trips_through_a_file(self, dataset, tmp_path):
        path = dataset.save(tmp_path / "questions.json")

        reloaded = Dataset.load(path)

        assert len(reloaded) == len(dataset)
        assert [q.id for q in reloaded] == [q.id for q in dataset]
        assert reloaded.chunk_size_chars == dataset.chunk_size_chars

    def test_grades_survive_the_round_trip_as_integers(self, dataset, tmp_path):
        path = dataset.save(tmp_path / "questions.json")

        reloaded = Dataset.load(path)

        assert reloaded.questions[0].relevance == {"x.md": 2}

    def test_the_saved_file_is_stable_for_diffing(self, dataset, tmp_path):
        """Two saves of the same dataset must be byte-identical, or review
        diffs fill with noise."""
        first = dataset.save(tmp_path / "a.json").read_text(encoding="utf-8")
        second = dataset.save(tmp_path / "b.json").read_text(encoding="utf-8")

        assert first == second


class TestDocumentCoverage:
    """Spread across the corpus is a status a human reads, not a gate this
    module enforces (spec.md: "spread across the corpus matters"; no
    threshold is chosen here — see ticket #12)."""

    @pytest.fixture
    def dataset(self) -> Dataset:
        return Dataset(
            questions=[
                Question(id="f1", question="a", stratum=Stratum.FACTUAL,
                          source_document="005_APOLLO_AND_DAPHNE.md"),
                Question(id="f2", question="b", stratum=Stratum.FACTUAL,
                          source_document="005_APOLLO_AND_DAPHNE.md"),
                Question(id="s1", question="c", stratum=Stratum.SYNTHESIS,
                          source_document="071_ORPHEUS_AND_EURYDICE.md"),
                Question(id="u1", question="d", stratum=Stratum.UNANSWERABLE),
            ]
        )

    def test_source_documents_deduplicates_and_excludes_unanswerable(self, dataset):
        """Two questions can share a source document (multi-hop siblings do);
        unanswerable questions carry no source_document by construction."""
        assert dataset.source_documents() == [
            "005_APOLLO_AND_DAPHNE.md",
            "071_ORPHEUS_AND_EURYDICE.md",
        ]

    def test_document_coverage_reports_distinct_count_against_corpus_size(self, dataset):
        assert dataset.document_coverage(corpus_size=164) == {
            "distinct_source_documents": 2,
            "corpus_size": 164,
            "fraction": round(2 / 164, 4),
        }

    def test_document_coverage_handles_a_zero_corpus_size(self, dataset):
        """Guards the division; corpus_size=0 has no real caller but should
        not raise."""
        assert dataset.document_coverage(corpus_size=0)["fraction"] == 0.0

    def test_to_dict_omits_coverage_when_corpus_size_is_not_supplied(self, dataset):
        assert "document_coverage" not in dataset.to_dict()

    def test_to_dict_includes_coverage_when_corpus_size_is_supplied(self, dataset):
        assert dataset.to_dict(corpus_size=164)["document_coverage"] == {
            "distinct_source_documents": 2,
            "corpus_size": 164,
            "fraction": round(2 / 164, 4),
        }
