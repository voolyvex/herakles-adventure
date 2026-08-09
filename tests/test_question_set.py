"""Acceptance checks for the committed question set (ticket #8).

Unlike test_dataset.py, which exercises Dataset/Question behaviour against
fixtures, this module asserts the actual committed eval_data/questions.json
meets #8's acceptance criteria — so a future edit that silently breaks one
of them fails CI instead of review.
"""

from __future__ import annotations

from collections import Counter

import pytest

from myth_eval.dataset import Dataset, Stratum, default_dataset_path


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return Dataset.load(default_dataset_path())


def test_the_dataset_has_roughly_forty_one_questions(dataset):
    assert 38 <= len(dataset) <= 44


def test_stratum_counts_match_the_spec_weighting(dataset):
    counts = Counter(q.stratum for q in dataset)

    assert counts[Stratum.FACTUAL] == 15
    assert counts[Stratum.SYNTHESIS] == 10
    assert counts[Stratum.ENTITY] == 8
    assert counts[Stratum.UNANSWERABLE] == 8


def test_question_ids_are_unique(dataset):
    ids = [q.id for q in dataset]

    assert len(ids) == len(set(ids))


def test_no_question_carries_relevance_labels_yet(dataset):
    """Grading is a separate ticket (#10) — this dataset must ship unlabelled."""
    for question in dataset:
        assert question.relevance == {}, f"{question.id} has relevance labels"


def test_every_in_corpus_question_has_a_source_document(dataset):
    for question in dataset:
        if question.stratum == Stratum.UNANSWERABLE:
            continue
        assert question.source_document, f"{question.id} is missing source_document"


def test_no_unanswerable_question_has_a_source_document(dataset):
    """Unanswerable questions are drawn from outside the corpus by construction."""
    for question in dataset.by_stratum(Stratum.UNANSWERABLE):
        assert question.source_document is None, f"{question.id} has a source_document"


def test_every_unanswerable_question_records_its_absence_verification(dataset):
    for question in dataset.by_stratum(Stratum.UNANSWERABLE):
        assert question.notes, f"{question.id} has no absence-verification notes"


def test_every_synthesis_question_records_its_additional_sources(dataset):
    """source_document only holds one file; multi-hop provenance lives in notes."""
    for question in dataset.by_stratum(Stratum.SYNTHESIS):
        assert question.notes, f"{question.id} has no multi-hop provenance in notes"


def test_chunker_provenance_is_recorded(dataset):
    assert dataset.chunk_size_chars > 0
    assert dataset.chunk_overlap_chars >= 0
    assert dataset.corpus_dir


def test_a_subset_of_questions_is_flagged_as_using_absent_vocabulary(dataset):
    """Acceptance criterion: some questions deliberately avoid source vocabulary,
    so the dense-vs-sparse comparison has something to discriminate on."""
    flagged = [
        q
        for q in dataset
        if q.notes and "vocabulary absent" in q.notes.lower()
    ]

    assert len(flagged) >= 3
