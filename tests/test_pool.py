"""The candidate pool generator: union across arms, deduplicated, reproducible.

Driven entirely through the fake retriever, so the whole suite runs with no
ChromaDB, no embedding model, no reranker and no network — the same seam the
runner tests use.

A note for anyone extending these tests: ``cli.build_fake_arms`` scripts its
responses from ``question.relevance``, and the committed question set is
unlabelled by design. Scripting a fake against the real dataset therefore
yields an empty pool, which is a property of the dataset rather than a bug in
the generator. These tests script their responses explicitly.
"""

from __future__ import annotations

from myth_eval.dataset import Dataset, Question, Stratum
from myth_eval.fakes import FakeRetriever
from myth_eval.pool import (
    EXCERPT_CHARS,
    CandidatePool,
    build_pool,
    default_pool_path,
)
from myth_eval.runner import POOL_DEPTH, evaluate_arm


def make_dataset(*questions: Question) -> Dataset:
    return Dataset(
        questions=list(questions),
        chunk_size_chars=1200,
        chunk_overlap_chars=200,
        embedding_model="BAAI/bge-small-en-v1.5",
    )


def factual(qid: str = "factual-01", text: str = "Who armed Perseus?") -> Question:
    return Question(id=qid, question=text, stratum=Stratum.FACTUAL)


def unanswerable(qid: str = "unanswerable-01") -> Question:
    return Question(
        id=qid,
        question="Which Norse god forged Mjolnir?",
        stratum=Stratum.UNANSWERABLE,
    )


def arms_for(dataset: Dataset, **scripts: dict) -> list:
    """Evaluate one fake arm per keyword, each scripted by question text."""
    return [
        evaluate_arm(FakeRetriever(name, responses=responses), dataset)
        for name, responses in scripts.items()
    ]


class TestUnionAcrossArms:
    """The criterion the whole ticket turns on."""

    def test_the_pool_is_the_union_of_every_arm_top_results(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(
            dataset,
            dense={question.question: ["a.md", "b.md"]},
            sparse={question.question: ["c.md"]},
            hybrid={question.question: ["b.md", "d.md"]},
        )

        pool = build_pool(arms, dataset)

        assert sorted(pool.gradeable[0].documents) == ["a.md", "b.md", "c.md", "d.md"]

    def test_a_document_only_one_arm_found_still_reaches_the_pool(self):
        """The fairness property: pooling must not favour the majority arm."""
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(
            dataset,
            dense={question.question: ["common.md"]},
            sparse={question.question: ["common.md"]},
            lone={question.question: ["only_this_arm_found_it.md"]},
        )

        pool = build_pool(arms, dataset)

        assert "only_this_arm_found_it.md" in pool.gradeable[0].documents

    def test_every_arm_is_recorded_in_provenance(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(
            dataset,
            dense={question.question: ["a.md"]},
            sparse={question.question: ["b.md"]},
        )

        pool = build_pool(arms, dataset)

        assert pool.arms == ["dense", "sparse"]
        assert pool.to_dict()["provenance"]["arms"] == ["dense", "sparse"]

    def test_an_arm_that_returned_nothing_contributes_nothing(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(
            dataset,
            dense={question.question: ["a.md"]},
            silent={},
        )

        pool = build_pool(arms, dataset)

        assert pool.gradeable[0].documents == ["a.md"]


class TestDeduplication:
    def test_a_document_found_by_three_arms_is_one_grading_decision(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(
            dataset,
            dense={question.question: ["shared.md"]},
            sparse={question.question: ["shared.md"]},
            hybrid={question.question: ["shared.md"]},
        )

        pool = build_pool(arms, dataset)

        assert pool.gradeable[0].documents == ["shared.md"]
        assert pool.candidate_count() == 1

    def test_several_chunks_of_one_document_collapse_to_one_candidate(self):
        """Labels key on the document, so its chunk count must not inflate the pool."""
        question = factual()
        dataset = make_dataset(question)
        # The fake returns one chunk per named document, so naming the same
        # document repeatedly is how an arm returning several of its chunks
        # reaches the pooler.
        arms = arms_for(
            dataset,
            dense={question.question: ["same.md", "same.md", "same.md", "other.md"]},
        )

        pool = build_pool(arms, dataset)

        assert pool.gradeable[0].documents == ["same.md", "other.md"]

    def test_deduplication_records_every_arm_that_found_the_document(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(
            dataset,
            dense={question.question: ["shared.md"]},
            sparse={question.question: ["shared.md"]},
        )

        pool = build_pool(arms, dataset)

        assert pool.gradeable[0].candidates[0].found_by == ["dense", "sparse"]

    def test_the_best_ranked_chunk_supplies_the_excerpt(self):
        """The labeller should read the strongest passage any arm found."""
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(
            dataset,
            weak={question.question: ["filler.md", "filler2.md", "target.md"]},
            strong={question.question: ["target.md"]},
        )

        pool = build_pool(arms, dataset)
        candidate = next(
            c for c in pool.gradeable[0].candidates if c.source_document == "target.md"
        )

        assert candidate.best_rank == 0
        assert candidate.chunk_id == "target.md_c0"

    def test_a_blank_source_document_is_not_offered_for_grading(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(dataset, dense={question.question: ["", "real.md"]})

        pool = build_pool(arms, dataset)

        assert pool.gradeable[0].documents == ["real.md"]


class TestShapedForGrading:
    def test_each_question_carries_its_text_alongside_its_candidates(self):
        question = factual(text="Who equipped Perseus with the winged shoes?")
        dataset = make_dataset(question)
        arms = arms_for(dataset, dense={question.question: ["033_PERSEUS.md"]})

        emitted = build_pool(arms, dataset).to_dict()["questions"][0]

        assert emitted["question"] == "Who equipped Perseus with the winged shoes?"
        assert emitted["stratum"] == Stratum.FACTUAL
        assert emitted["candidates"][0]["source_document"] == "033_PERSEUS.md"

    def test_every_candidate_has_an_empty_grade_slot_to_fill_in(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(dataset, dense={question.question: ["a.md", "b.md"]})

        emitted = build_pool(arms, dataset).to_dict()["questions"][0]

        assert [c["grade"] for c in emitted["candidates"]] == [None, None]

    def test_candidates_carry_the_text_a_labeller_reads(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(dataset, dense={question.question: ["a.md"]})

        emitted = build_pool(arms, dataset).to_dict()["questions"][0]

        assert emitted["candidates"][0]["excerpt"]

    def test_a_long_excerpt_is_trimmed_to_a_readable_window(self):
        question = factual()
        dataset = make_dataset(question)
        # A chunk longer than the window, built directly through the protocol
        # type: the fake's own passages are short by construction.
        from myth_eval.retrieval import RetrievedItem

        arm = evaluate_arm(FakeRetriever("dense"), dataset)
        arm.outcomes[0].items = [
            RetrievedItem(
                chunk_id="long_c0",
                source_document="long.md",
                text="word " * 500,
                score=1.0,
                rank=0,
            )
        ]

        pool = build_pool([arm], dataset)
        excerpt = pool.gradeable[0].candidates[0].excerpt

        assert len(excerpt) <= EXCERPT_CHARS + 3
        assert excerpt.endswith("...")

    def test_the_grading_view_hides_arm_and_score(self):
        """Grading blind to arm is what stops the gold set favouring one arm."""
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(dataset, dense={question.question: ["a.md"]})

        emitted = build_pool(arms, dataset).to_dict()["questions"][0]
        candidate = emitted["candidates"][0]

        assert "found_by" not in candidate
        assert "score" not in candidate
        assert "best_rank" not in candidate

    def test_arm_attribution_survives_as_a_diagnostic(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(dataset, dense={question.question: ["a.md"]})

        diagnostics = build_pool(arms, dataset).to_dict()["diagnostics"]

        attributed = diagnostics["attribution"][0]["candidates"][0]
        assert attributed["found_by"] == ["dense"]

    def test_the_sheet_states_the_grading_scale(self):
        dataset = make_dataset(factual())
        emitted = build_pool([], dataset).to_dict()

        assert emitted["grade_labels"] == {
            "0": "irrelevant",
            "1": "related context",
            "2": "directly answers",
        }
        assert "three-point scale" in emitted["instructions"]


class TestUnanswerableQuestions:
    """The rubric is explicit: these are skipped, not graded."""

    def test_an_unanswerable_question_is_not_emitted_as_a_grading_task(self):
        answerable, skipped = factual(), unanswerable()
        dataset = make_dataset(answerable, skipped)
        arms = arms_for(
            dataset,
            dense={
                answerable.question: ["a.md"],
                skipped.question: ["near_miss.md"],
            },
        )

        pool = build_pool(arms, dataset)

        assert [p.question_id for p in pool.gradeable] == ["factual-01"]
        assert "unanswerable-01" not in {
            q["id"] for q in pool.to_dict()["questions"]
        }

    def test_what_the_arms_returned_for_it_is_still_reported(self):
        skipped = unanswerable()
        dataset = make_dataset(skipped)
        arms = arms_for(dataset, dense={skipped.question: ["near_miss.md"]})

        diagnostics = build_pool(arms, dataset).to_dict()["diagnostics"]

        assert diagnostics["unanswerable"][0]["id"] == "unanswerable-01"
        assert (
            diagnostics["unanswerable"][0]["candidates"][0]["source_document"]
            == "near_miss.md"
        )

    def test_its_candidates_are_not_counted_as_grading_decisions(self):
        answerable, skipped = factual(), unanswerable()
        dataset = make_dataset(answerable, skipped)
        arms = arms_for(
            dataset,
            dense={
                answerable.question: ["a.md"],
                skipped.question: ["x.md", "y.md", "z.md"],
            },
        )

        pool = build_pool(arms, dataset)

        assert pool.candidate_count() == 1
        assert pool.to_dict()["summary"]["questions_not_graded"] == 1


class TestPoolDepth:
    def test_pooling_takes_the_top_ten_by_default(self):
        assert POOL_DEPTH == 10

        question = factual()
        dataset = make_dataset(question)
        documents = [f"doc{i:02d}.md" for i in range(15)]
        arms = arms_for(dataset, dense={question.question: documents})

        pool = build_pool(arms, dataset)

        assert pool.gradeable[0].documents == documents[:POOL_DEPTH]

    def test_a_shallower_depth_is_recorded_rather_than_silent(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(dataset, dense={question.question: ["a.md", "b.md", "c.md"]})

        pool = build_pool(arms, dataset, depth=2)

        assert pool.gradeable[0].documents == ["a.md", "b.md"]
        assert pool.to_dict()["provenance"]["pool_depth"] == 2

    def test_a_deeper_retrieval_run_does_not_widen_the_pool(self, tmp_path):
        """--k sets retrieval depth; it must not push the pool past ten."""
        from myth_eval.cli import main

        question = factual()
        dataset = make_dataset(question)
        # Labelled, because build_fake_arms scripts its responses from the
        # relevance map — an unlabelled question yields an empty pool.
        question.relevance = {f"doc{i:02d}.md": 2 for i in range(15)}
        dataset_path = tmp_path / "questions.json"
        dataset.save(dataset_path)

        pool_path = tmp_path / "pool.json"
        exit_code = main(
            [
                "--fake",
                "--k",
                "20",
                "--dataset",
                str(dataset_path),
                "--output",
                str(tmp_path / "results.json"),
                "--pool-output",
                str(pool_path),
            ]
        )

        emitted = CandidatePool.load(pool_path)
        assert exit_code == 0
        assert emitted["provenance"]["pool_depth"] == POOL_DEPTH
        assert len(emitted["questions"][0]["candidates"]) == POOL_DEPTH


class TestReproducibility:
    def test_the_same_inputs_produce_byte_identical_output(self, tmp_path):
        question = factual()
        dataset = make_dataset(question)
        scripts = {
            "dense": {question.question: ["b.md", "a.md"]},
            "sparse": {question.question: ["c.md", "a.md"]},
        }

        first = build_pool(arms_for(dataset, **scripts), dataset).save(
            tmp_path / "one.json"
        )
        second = build_pool(arms_for(dataset, **scripts), dataset).save(
            tmp_path / "two.json"
        )

        assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")

    def test_arm_order_changes_neither_the_ordering_nor_the_excerpts(self):
        """Reordering the arms must not reorder the sheet or swap a passage.

        The excerpt matters as much as the ordering: two arms returning the
        same document at the same rank must not hand the labeller a different
        passage depending on which arm was processed first.
        """
        question = factual()
        dataset = make_dataset(question)
        scripts = {
            "dense": {question.question: ["tied.md", "b.md", "a.md"]},
            "sparse": {question.question: ["tied.md", "c.md"]},
        }

        forward = build_pool(arms_for(dataset, **scripts), dataset)
        backward = build_pool(list(reversed(arms_for(dataset, **scripts))), dataset)

        def view(pool):
            return [
                (c.source_document, c.chunk_id, c.excerpt)
                for c in pool.gradeable[0].candidates
            ]

        assert view(forward) == view(backward)

    def test_ties_on_rank_break_on_document_name(self):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(
            dataset,
            dense={question.question: ["zeta.md"]},
            sparse={question.question: ["alpha.md"]},
        )

        pool = build_pool(arms, dataset)

        assert pool.gradeable[0].documents == ["alpha.md", "zeta.md"]

    def test_provenance_records_the_chunking_regime_the_pool_was_built_under(self):
        dataset = make_dataset(factual())

        provenance = build_pool([], dataset).to_dict()["provenance"]

        assert provenance["chunk_size_chars"] == 1200
        assert provenance["chunk_overlap_chars"] == 200
        assert provenance["embedding_model"] == "BAAI/bge-small-en-v1.5"


class TestPersistence:
    def test_a_saved_pool_round_trips_as_json(self, tmp_path):
        question = factual()
        dataset = make_dataset(question)
        arms = arms_for(dataset, dense={question.question: ["a.md"]})

        path = build_pool(arms, dataset).save(tmp_path / "pool.json")
        loaded = CandidatePool.load(path)

        assert loaded["questions"][0]["candidates"][0]["source_document"] == "a.md"

    def test_the_default_path_sits_beside_the_dataset(self):
        assert default_pool_path().name == "candidate_pool.json"
        assert default_pool_path().parent.name == "eval_data"

    def test_the_summary_counts_what_the_sitting_will_cost(self, tmp_path):
        one, two = factual("factual-01"), factual("factual-02", "Another question?")
        dataset = make_dataset(one, two)
        arms = arms_for(
            dataset,
            dense={one.question: ["a.md", "b.md"], two.question: ["b.md"]},
        )

        summary = build_pool(arms, dataset).to_dict()["summary"]

        assert summary["questions_to_grade"] == 2
        assert summary["grading_decisions"] == 3
        # b.md is pooled for both questions but is one document.
        assert summary["distinct_documents"] == 2


class TestNoHeavyDependencies:
    def test_the_pool_suite_loads_no_models_or_stores(self):
        """Pooling rides the same seam the runner does."""
        import sys

        for forbidden in ("chromadb", "sentence_transformers", "flashrank", "torch"):
            assert forbidden not in sys.modules, (
                f"{forbidden} was imported; the pool suite must stay free of it"
            )
