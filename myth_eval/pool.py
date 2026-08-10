"""The candidate pool: what a human actually sits down and grades.

Pooled judging. Every arm runs over the question set, the union of their top-10
results is taken per question, deduplicated by source document, and emitted as
one prepared labelling list. Anything outside the pool scores zero.

Pooling is chosen over grading only the current retriever's output because the
alternative biases the gold set toward today's system: a genuinely better
future retriever would surface good documents nobody had labelled and be
penalised for it. That fairness is the whole point, and it is what makes the
baseline meaningful to a backend that does not exist yet.

Two properties of the emitted artefact are load-bearing, and both come straight
from ``grading_rubric.md``:

**Candidates are identified by source document.** A document appears once per
question no matter how many of its chunks any arm returned. Labels key on the
document because chunk IDs are a function of the chunker's size and overlap, so
labelling against them means a re-chunk silently rots the dataset. One
representative chunk's text is carried along as the window into that document —
the labeller reads it to decide, but grades the document.

**The grading view is blind to arm and score.** The rubric puts "retriever
score or rank" under what is explicitly *not* graded against, because knowing
which arm surfaced a candidate biases the gold set toward that arm — the exact
bias pooling exists to remove. Arm attribution is computed, because it is a
genuinely useful diagnostic, but it is kept in a separate ``diagnostics``
section rather than sitting next to the text a human is reading.

Blindness here means the *ordering* too, not just the absent key. The graded
candidate list is sorted by source document alone, so where a candidate sits on
the page says nothing about how any arm ranked it. Sorting best-first would put
the rank back in as ordinal position and hand the labeller the system's
favourites to read first — which is also the comparison the rubric forbids,
since it asks for each candidate to be graded independently of the others.

Unanswerable questions are not emitted as grading tasks at all. The rubric is
explicit that they are skipped and that writing 0s into their relevance map
misrepresents "grading does not apply here" as "grading happened here". Their
pooled candidates are still reported, under diagnostics, because what an arm
returns for a question with no answer is worth seeing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from myth_eval.dataset import GRADE_LABELS, Dataset, Question
from myth_eval.runner import POOL_DEPTH, ArmResult

logger = logging.getLogger(__name__)

__all__ = [
    "EXCERPT_CHARS",
    "PooledCandidate",
    "QuestionPool",
    "CandidatePool",
    "build_pool",
    "default_pool_path",
]

# How much of a chunk travels into the grading sheet. Enough to decide from,
# short enough that a few hundred candidates stay readable in one sitting; the
# rubric tells the labeller to consult the source document when a chunk cuts
# off mid-fact, so this is a window, never the evidence itself.
EXCERPT_CHARS = 600

GRADING_INSTRUCTIONS = (
    "Grade each candidate against its source document as a whole, on the "
    "three-point scale: 2 directly answers, 1 related context, 0 irrelevant. "
    "Walk 2, then 1, then 0, and stop at the first that fits. Grade "
    "candidates independently of each other, and when in doubt default down a "
    "grade rather than up. See .scratch/rag-evaluation-harness/grading_rubric.md."
)


def default_pool_path() -> Path:
    """Where a generated candidate pool is written by default.

    Not a committed file, and deliberately so: generating a real pool needs a
    built index, and the only pool that can be produced without one comes from
    scripted fake arms — which sitting at this path would read as genuine.
    """
    return Path(__file__).resolve().parent.parent / "eval_data" / "candidate_pool.json"


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """One chunk's text, trimmed to a readable window on a whitespace boundary."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit]
    boundary = cut.rfind(" ")
    if boundary > limit // 2:
        cut = cut[:boundary]
    return cut + "..."


@dataclass
class PooledCandidate:
    """One source document offered for grading against one question.

    Attributes:
        source_document: The corpus file. This is the identity the grade
            attaches to, and the reason a re-chunk does not invalidate labels.
        excerpt: Text from the best-ranked chunk of this document across all
            arms — the window the labeller reads to decide.
        chunk_id: That chunk's identifier. Diagnostic: it lets someone trace a
            candidate back to a specific chunk, but grades never key on it.
        best_rank: The best (lowest) zero-based rank this document reached in
            any arm. Chooses which chunk supplies the excerpt, and is reported
            under diagnostics. It orders nothing in the graded view: the sheet
            is sorted by document name precisely so that ordinal position
            carries no rank information.
        found_by: Which arms surfaced this document. Diagnostic only.
    """

    source_document: str
    excerpt: str
    chunk_id: str = ""
    best_rank: int = 0
    found_by: List[str] = field(default_factory=list)

    def to_grading_dict(self) -> Dict[str, Any]:
        """The view a human grades from: document, text, and a slot for a grade.

        Carries no arm attribution and no score. That omission is the point —
        see this module's docstring.
        """
        return {
            "source_document": self.source_document,
            "excerpt": self.excerpt,
            "grade": None,
        }


@dataclass
class QuestionPool:
    """The deduplicated candidate set for one question."""

    question_id: str
    question: str
    stratum: str
    candidates: List[PooledCandidate] = field(default_factory=list)

    @property
    def documents(self) -> List[str]:
        return [candidate.source_document for candidate in self.candidates]

    def to_grading_dict(self) -> Dict[str, Any]:
        return {
            "id": self.question_id,
            "question": self.question,
            "stratum": self.stratum,
            "candidates": [c.to_grading_dict() for c in self.candidates],
        }

    def to_diagnostic_dict(self) -> Dict[str, Any]:
        return {
            "id": self.question_id,
            "stratum": self.stratum,
            "candidates": [
                {
                    "source_document": c.source_document,
                    "chunk_id": c.chunk_id,
                    "best_rank": c.best_rank,
                    "found_by": list(c.found_by),
                }
                for c in self.candidates
            ],
        }


@dataclass
class CandidatePool:
    """Every question's pool, plus the provenance that makes it reproducible.

    ``gradeable`` and ``diagnostic_only`` partition the questions. The split is
    not cosmetic: emitting unanswerable questions as grading tasks invites 0s
    into a relevance map that is empty *by construction*, which would read to a
    later maintainer as grading having happened where it does not apply.
    """

    gradeable: List[QuestionPool] = field(default_factory=list)
    diagnostic_only: List[QuestionPool] = field(default_factory=list)
    arms: List[str] = field(default_factory=list)
    depth: int = POOL_DEPTH
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def unique_documents(self) -> List[str]:
        """Every distinct document in the pool, across all questions."""
        return sorted(
            {
                candidate.source_document
                for pool in self.gradeable + self.diagnostic_only
                for candidate in pool.candidates
            }
        )

    def candidate_count(self) -> int:
        """Total grading decisions: one per (question, document) pair."""
        return sum(len(pool.candidates) for pool in self.gradeable)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instructions": GRADING_INSTRUCTIONS,
            "grade_labels": {str(k): v for k, v in sorted(GRADE_LABELS.items())},
            "provenance": {
                "pool_depth": self.depth,
                "arms": list(self.arms),
                **self.provenance,
            },
            "summary": {
                "questions_to_grade": len(self.gradeable),
                "grading_decisions": self.candidate_count(),
                "distinct_documents": len(self.unique_documents),
                "questions_not_graded": len(self.diagnostic_only),
            },
            "questions": [pool.to_grading_dict() for pool in self.gradeable],
            "diagnostics": {
                "_note": (
                    "Arm attribution and ranks live here, apart from the "
                    "grading view, so grading stays blind to which arm found "
                    "a candidate. Unanswerable questions are listed here and "
                    "are not graded: their empty relevance map is their ground "
                    "truth, so what an arm returns for them is a pooling "
                    "diagnostic, not a labelling task."
                ),
                "unanswerable": [
                    pool.to_diagnostic_dict() for pool in self.diagnostic_only
                ],
                "attribution": [
                    pool.to_diagnostic_dict() for pool in self.gradeable
                ],
            },
        }

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path is not None else default_pool_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path) -> Dict[str, Any]:
        """Load a pool file as a plain dict, for inspection and for tests."""
        return json.loads(Path(path).read_text(encoding="utf-8"))


def _outcomes_by_question(
    arms: Sequence[ArmResult],
) -> List[tuple]:
    """Index each arm's outcomes by question id, once for the whole run."""
    return [
        (arm.name, {outcome.question_id: outcome for outcome in arm.outcomes})
        for arm in arms
    ]


def _pool_one_question(
    question: Question,
    indexed_arms: Sequence[tuple],
    depth: int,
) -> QuestionPool:
    """Union one question's results across arms, deduplicated by document.

    A document seen by several arms, or by one arm at several chunks, becomes
    exactly one candidate. The representative chunk and the recorded rank come
    from wherever the document ranked best, so the labeller reads the strongest
    passage any arm found rather than an arbitrary one.
    """
    merged: Dict[str, PooledCandidate] = {}

    for arm_name, outcomes in indexed_arms:
        outcome = outcomes.get(question.id)
        if outcome is None:
            continue

        for item in outcome.items[:depth]:
            document = item.source_document
            # normalise_hit defaults a missing source to "", which would emit a
            # blank, ungradeable candidate. Drop it rather than ask a human to
            # grade nothing.
            if not document:
                continue

            existing = merged.get(document)
            if existing is None:
                merged[document] = PooledCandidate(
                    source_document=document,
                    excerpt=_excerpt(item.text),
                    chunk_id=item.chunk_id,
                    best_rank=item.rank,
                    found_by=[arm_name],
                )
                continue

            if arm_name not in existing.found_by:
                existing.found_by.append(arm_name)
            # A better-ranked chunk of the same document replaces the window,
            # so the excerpt shown is the strongest passage found. Chunk id
            # breaks a rank tie, making the choice total: without it, two arms
            # returning the same document at the same rank would hand the
            # labeller a different passage depending on arm order.
            if (item.rank, item.chunk_id) < (existing.best_rank, existing.chunk_id):
                existing.best_rank = item.rank
                existing.excerpt = _excerpt(item.text)
                existing.chunk_id = item.chunk_id

    # Document name alone. Sorting on best rank would have put the retriever's
    # ranking back into the sheet as ordinal position — omitting the key from
    # the graded view moves the information into the sequence rather than
    # removing it, and a best-first sheet invites the cross-candidate
    # comparison the rubric forbids. source_document is the dedup key and so is
    # already unique within a question, which keeps this a total order and the
    # pool reproducible run to run.
    candidates = sorted(
        merged.values(), key=lambda candidate: candidate.source_document
    )
    for candidate in candidates:
        candidate.found_by.sort()

    return QuestionPool(
        question_id=question.id,
        question=question.question,
        stratum=question.stratum,
        candidates=candidates,
    )


def build_pool(
    arms: Sequence[ArmResult],
    dataset: Dataset,
    depth: int = POOL_DEPTH,
) -> CandidatePool:
    """Build the deduplicated candidate pool from every arm's results.

    Args:
        arms: The evaluated arms, carrying their retained per-question outcomes.
        dataset: The question set the arms ran over.
        depth: How deep into each arm's ranking to pool. Defaults to the
            pooling depth the spec fixes at 10. Pooling shallower is allowed —
            a smoke test is a legitimate reason — but it is warned about, and
            the recorded ``pool_depth`` keeps it visible in the artefact.

    Returns:
        The pool, partitioned into questions to grade and questions reported
        for diagnostics only.
    """
    if depth < POOL_DEPTH:
        # Not a refusal: a shallow pool is fine for a smoke test. But it must
        # not be mistaken for the real one, because everything outside the pool
        # scores zero — so a pool built shallow permanently zeroes documents a
        # conforming pool would have carried, and no later run can detect it.
        logger.warning(
            "Pooling at depth %d, below the spec's pooling depth of %d. This "
            "pool does not conform to the spec: documents a full-depth pool "
            "would have contained will score zero, and nothing downstream can "
            "detect that. Use it for a smoke test, not for grading.",
            depth,
            POOL_DEPTH,
        )

    gradeable: List[QuestionPool] = []
    diagnostic_only: List[QuestionPool] = []
    indexed_arms = _outcomes_by_question(arms)

    for question in dataset:
        pool = _pool_one_question(question, indexed_arms, depth)
        if question.is_unanswerable:
            diagnostic_only.append(pool)
        else:
            gradeable.append(pool)

    return CandidatePool(
        gradeable=gradeable,
        diagnostic_only=diagnostic_only,
        arms=[arm.name for arm in arms],
        depth=depth,
        provenance={
            "chunk_size_chars": dataset.chunk_size_chars,
            "chunk_overlap_chars": dataset.chunk_overlap_chars,
            "corpus_dir": dataset.corpus_dir,
            "embedding_model": dataset.embedding_model,
            "question_count": len(dataset),
        },
    )
