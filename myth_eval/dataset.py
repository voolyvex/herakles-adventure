"""The evaluation dataset: questions, strata, and graded relevance labels.

One version-controlled JSON file holds the whole dataset, so two runs a month
apart are comparable and the gold standard is diffable in review.

Relevance labels attach to **source documents**, not chunk identifiers. Chunk
IDs are deterministic but are a function of the chunker's size and overlap
settings, so labelling against them means any re-chunk silently rots the
dataset. Chunker parameters are recorded as provenance so a later reader knows
which chunking regime the labels were gathered under.

Grades are a three-point scale — 0 irrelevant, 1 related context, 2 directly
answers. Three levels rather than four because a single labeller across ~41
questions will not apply finer distinctions consistently, and inconsistency
becomes noise in the metric the harness exists to make trustworthy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "Stratum",
    "Question",
    "Dataset",
    "GRADE_LABELS",
    "default_dataset_path",
]

# The four strata. Deliberately uneven in size, weighted toward what will
# actually be tuned against.
class Stratum:
    FACTUAL = "factual"
    SYNTHESIS = "synthesis"
    ENTITY = "entity"
    UNANSWERABLE = "unanswerable"

    ALL = (FACTUAL, SYNTHESIS, ENTITY, UNANSWERABLE)


GRADE_LABELS: Mapping[int, str] = {
    0: "irrelevant",
    1: "related context",
    2: "directly answers",
}


def default_dataset_path() -> Path:
    """The committed dataset file."""
    return Path(__file__).resolve().parent.parent / "eval_data" / "questions.json"


@dataclass
class Question:
    """One evaluation question and its graded ground truth.

    Attributes:
        id: Stable identifier, referenced by the candidate pool and by results.
        question: The query text put to the retriever.
        stratum: One of :class:`Stratum`.
        relevance: Map of source document to grade (0/1/2). Empty for
            unanswerable questions, where returning nothing is correct, and
            empty before the manual grading pass has run.
        source_document: For in-corpus questions, the document the question was
            generated from. Provenance only — it is not treated as ground truth,
            because pooled judging may well find better documents elsewhere.
        notes: Free-text, e.g. recording that a question deliberately uses
            vocabulary absent from its source passage.
    """

    id: str
    question: str
    stratum: str
    relevance: Dict[str, int] = field(default_factory=dict)
    source_document: Optional[str] = None
    notes: Optional[str] = None

    @property
    def is_unanswerable(self) -> bool:
        return self.stratum == Stratum.UNANSWERABLE

    @property
    def is_labelled(self) -> bool:
        """Whether grading has happened for this question.

        An unanswerable question is labelled by construction: the correct
        behaviour is returning nothing, so an empty relevance map *is* its
        ground truth rather than an absence of one.
        """
        return self.is_unanswerable or bool(self.relevance)

    def relevant_documents(self, threshold: int = 1) -> List[str]:
        """Source documents graded at or above ``threshold``."""
        return sorted(
            document
            for document, grade in self.relevance.items()
            if int(grade) >= threshold
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "stratum": self.stratum,
            "relevance": dict(sorted(self.relevance.items())),
        }
        if self.source_document:
            data["source_document"] = self.source_document
        if self.notes:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Question":
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            stratum=str(data["stratum"]),
            relevance={
                str(document): int(grade)
                for document, grade in (data.get("relevance") or {}).items()
            },
            source_document=data.get("source_document"),
            notes=data.get("notes"),
        )


@dataclass
class Dataset:
    """The full question set plus the provenance of how it was gathered."""

    questions: List[Question] = field(default_factory=list)
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    corpus_dir: str = "lore_chunks"
    embedding_model: Optional[str] = None
    notes: Optional[str] = None

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def by_stratum(self, stratum: str) -> List[Question]:
        return [q for q in self.questions if q.stratum == stratum]

    def stratum_counts(self) -> Dict[str, int]:
        return {
            stratum: len(self.by_stratum(stratum))
            for stratum in Stratum.ALL
            if self.by_stratum(stratum)
        }

    def source_documents(self) -> List[str]:
        """Distinct source documents referenced by in-corpus questions.

        Unanswerable questions carry no ``source_document`` by construction
        (spec.md: they are drawn from mythologies outside the corpus) and are
        excluded. This is spread over *questions' declared source*, not over
        the pooled relevance labels — the pool (#9) may touch more documents
        than this once populated, since pooling draws from every arm's
        results, not just each question's origin chunk.
        """
        return sorted(
            {q.source_document for q in self.questions if q.source_document}
        )

    def document_coverage(self, corpus_size: int) -> Dict[str, Any]:
        """How much of the corpus the question set's declared sources touch.

        Not a gate. Spec.md says spread across the corpus matters more than
        depth on any one document, but nothing enforces that today — this is
        the visibility the harness offers a human deciding whether coverage
        is thin. ``corpus_size`` is a caller-supplied count (e.g. the number
        of files under ``lore_chunks/``) rather than something this module
        discovers itself, so ``Dataset`` stays free of filesystem I/O.
        """
        documents = self.source_documents()
        return {
            "distinct_source_documents": len(documents),
            "corpus_size": corpus_size,
            "fraction": round(len(documents) / corpus_size, 4) if corpus_size else 0.0,
        }

    @property
    def is_labelled(self) -> bool:
        """Whether every question has ground truth, i.e. grading has happened."""
        return bool(self.questions) and all(q.is_labelled for q in self.questions)

    def to_dict(self, corpus_size: Optional[int] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            # Provenance: which chunking regime the labels were gathered under.
            # Labels attach to source documents so they survive re-chunking, but
            # the pool that was graded was produced under these settings.
            "provenance": {
                "chunk_size_chars": self.chunk_size_chars,
                "chunk_overlap_chars": self.chunk_overlap_chars,
                "corpus_dir": self.corpus_dir,
            },
            "stratum_counts": self.stratum_counts(),
            "questions": [q.to_dict() for q in self.questions],
        }
        if self.embedding_model:
            data["provenance"]["embedding_model"] = self.embedding_model
        if corpus_size is not None:
            # Status only, not a gate: see document_coverage(). A floor for
            # this number, if one gets adopted, is decided in #12 alongside
            # the nDCG regression tolerance, not enforced here.
            data["document_coverage"] = self.document_coverage(corpus_size)
        if self.notes:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dataset":
        provenance = data.get("provenance") or {}
        return cls(
            questions=[Question.from_dict(q) for q in data.get("questions", [])],
            chunk_size_chars=int(provenance.get("chunk_size_chars", 1200)),
            chunk_overlap_chars=int(provenance.get("chunk_overlap_chars", 200)),
            corpus_dir=str(provenance.get("corpus_dir", "lore_chunks")),
            embedding_model=provenance.get("embedding_model"),
            notes=data.get("notes"),
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Dataset":
        target = path if path is not None else default_dataset_path()
        return cls.from_dict(json.loads(Path(target).read_text(encoding="utf-8")))

    def save(self, path: Optional[Path] = None, corpus_size: Optional[int] = None) -> Path:
        target = Path(path) if path is not None else default_dataset_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(corpus_size), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target
