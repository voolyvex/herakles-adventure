"""Ranking-quality metrics, as pure functions.

Ranked source-document identifiers plus a grade map in, a number out. No I/O,
no models, no vector store, no network — so these can be tested against
hand-computed values in milliseconds.

This is the primary test seam of the harness. If the mathematics here is wrong
then every number the harness produces is fiction, the committed baseline
included.

Relevance is **graded** on three levels:

===== ====================
Grade Meaning
===== ====================
0     irrelevant
1     related context
2     directly answers
===== ====================

Three levels rather than four because a single labeller across ~41 questions
will not apply finer distinctions consistently, and inconsistency becomes noise
in the metric the harness exists to make trustworthy.

Documents absent from the grade map score **zero**. That is pooled judging's
defined behaviour on unseen results, not an error case: the graded pool is the
union of all arms' top-10, and anything outside it is unjudged.

nDCG is the headline metric because it is the only one here that reads the
three-point scale — Recall and MRR are binary-relevance metrics and collapse
grades 1 and 2 together.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "dcg",
    "ndcg_at_k",
    "recall_at_k",
    "mrr_at_k",
    "unanswerable_precision",
    "percentile",
    "RELEVANT_THRESHOLD",
]

# A document counts as "relevant" for the binary metrics (Recall, MRR) when its
# grade is at least this. Grade 1 is related context, which is a hit for
# "did we find anything useful at all" — the question those metrics answer.
RELEVANT_THRESHOLD = 1


def _grade(document: str, grades: Mapping[str, int]) -> int:
    """Return the graded relevance of ``document``, defaulting to 0.

    Unjudged documents score zero by construction. This is pooled judging's
    defined behaviour, so a missing key is normal rather than exceptional.
    """
    try:
        return int(grades.get(document, 0))
    except (TypeError, ValueError):
        return 0


def dcg(gains: Iterable[float]) -> float:
    """Discounted cumulative gain over an ordered sequence of gains.

    Uses the exponential-gain formulation, ``(2**g - 1) / log2(rank + 1)``,
    which is the standard for graded relevance: it rewards a grade-2 document
    disproportionately more than a grade-1 one, matching the intent that
    "directly answers" is worth substantially more than "related context".

    Args:
        gains: Relevance grades in rank order, best position first.

    Returns:
        The discounted cumulative gain.
    """
    return sum(
        (2.0 ** float(gain) - 1.0) / math.log2(rank + 2)
        for rank, gain in enumerate(gains)
    )


def ndcg_at_k(
    ranked_documents: Sequence[str],
    grades: Mapping[str, int],
    k: int,
) -> float:
    """Normalised discounted cumulative gain at ``k``.

    The headline metric. Reads the full three-point grade scale, so a
    configuration returning a grade-2 document at rank 1 scores better than one
    returning it at rank 9.

    Normalisation is against the *ideal* ranking: every judged document sorted
    by grade descending, truncated to ``k``. So the score is 1.0 when the arm
    returns the best available documents in the best available order.

    Args:
        ranked_documents: Source-document identifiers in rank order, best
            first. Duplicates are collapsed, keeping the earliest occurrence,
            since a document cannot be relevant twice.
        grades: Map of document identifier to grade. Missing keys score 0.
        k: Cutoff. May exceed the number of results or the number of relevant
            documents.

    Returns:
        A value in [0.0, 1.0]. Returns 0.0 when no relevant document exists,
        which is the conventional treatment of an undefined ideal — for
        questions where returning nothing is correct, use
        :func:`unanswerable_precision` instead.
    """
    if k <= 0:
        return 0.0

    retrieved = _deduplicate(ranked_documents)[:k]
    actual = dcg(_grade(document, grades) for document in retrieved)

    ideal_gains = sorted(
        (int(grade) for grade in grades.values() if int(grade) > 0),
        reverse=True,
    )[:k]
    ideal = dcg(ideal_gains)

    if ideal == 0.0:
        # No relevant document exists for this question, so there is no ideal
        # ranking to normalise against and nDCG is undefined. Report 0.0.
        return 0.0

    return actual / ideal


def recall_at_k(
    ranked_documents: Sequence[str],
    grades: Mapping[str, int],
    k: int,
    threshold: int = RELEVANT_THRESHOLD,
) -> float:
    """Fraction of relevant documents found in the top ``k``.

    A binary-relevance metric: it cannot read the three-point scale and treats
    every grade at or above ``threshold`` as equally relevant. Reported as the
    legible secondary figure for readers who have not used nDCG.

    Args:
        ranked_documents: Source-document identifiers in rank order.
        grades: Map of document identifier to grade. Missing keys score 0.
        k: Cutoff.
        threshold: Minimum grade counting as relevant.

    Returns:
        A value in [0.0, 1.0]. Returns 0.0 when no document is relevant, since
        recall is undefined with an empty denominator.
    """
    if k <= 0:
        return 0.0

    relevant = {
        document
        for document, grade in grades.items()
        if _grade(document, grades) >= threshold
    }
    if not relevant:
        return 0.0

    retrieved = set(_deduplicate(ranked_documents)[:k])
    return len(retrieved & relevant) / len(relevant)


def mrr_at_k(
    ranked_documents: Sequence[str],
    grades: Mapping[str, int],
    k: int = 10,
    threshold: int = RELEVANT_THRESHOLD,
) -> float:
    """Reciprocal rank of the first relevant document within the top ``k``.

    Answers "how quickly does something useful appear". Binary-relevance, like
    Recall: grades 1 and 2 both count as a hit.

    Args:
        ranked_documents: Source-document identifiers in rank order.
        grades: Map of document identifier to grade. Missing keys score 0.
        k: Cutoff.
        threshold: Minimum grade counting as relevant.

    Returns:
        ``1 / rank`` of the first relevant document using one-based ranks, or
        0.0 if none appears within ``k``.
    """
    if k <= 0:
        return 0.0

    for position, document in enumerate(_deduplicate(ranked_documents)[:k], start=1):
        if _grade(document, grades) >= threshold:
            return 1.0 / position
    return 0.0


def unanswerable_precision(
    ranked_documents: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    confidence_threshold: float = 0.0,
) -> float:
    """Score an unanswerable question, where returning nothing is correct.

    Ordinary recall is undefined here: there is no relevant document to find,
    so the denominator is empty. The measure is instead precision at a
    confidence threshold — the system is correct when it abstains, meaning it
    returns nothing, or returns only results it scores below the threshold.

    Args:
        ranked_documents: Source-document identifiers the arm returned.
        scores: Corresponding retriever scores, aligned with
            ``ranked_documents``. When omitted, any returned result counts as a
            confident answer.
        confidence_threshold: Results scoring at or above this are treated as
            confident answers. With the default of 0.0 and scores supplied,
            only strictly-positive scores count as confident.

    Returns:
        1.0 when the system correctly abstained, 0.0 when it confidently
        returned something.
    """
    if not ranked_documents:
        return 1.0

    if scores is None:
        # No scores available: any returned result is treated as an answer.
        return 0.0

    confident = [
        score
        for score in scores[: len(ranked_documents)]
        if float(score) > confidence_threshold
    ]
    return 0.0 if confident else 1.0


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile, used for p50 and p95 latency.

    Kept here rather than pulling in numpy so the metric module stays free of
    heavy dependencies and runs in a minimal environment.

    Args:
        values: Observations. Need not be sorted.
        fraction: Percentile as a fraction, e.g. 0.5 for p50, 0.95 for p95.

    Returns:
        The interpolated percentile, or 0.0 for an empty input.
    """
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = fraction * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[int(position)]

    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _deduplicate(documents: Sequence[str]) -> List[str]:
    """Collapse repeats, keeping the earliest occurrence.

    Several chunks can share a source document, and labels attach to source
    documents, so the same identifier can legitimately appear more than once in
    one arm's ranking. A document cannot be relevant twice: counting it twice
    would inflate DCG above the ideal and let nDCG exceed 1.0.
    """
    seen = set()
    unique: List[str] = []
    for document in documents:
        if document not in seen:
            seen.add(document)
            unique.append(document)
    return unique
