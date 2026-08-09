"""Throwaway triage tool for ticket #8's paraphrase pass.

Scores each in-corpus question's lexical containment against its source
passage: |question_content_words ∩ source_content_words| / |question_content_words|,
stopwords stripped. High score = question still echoes the source's wording
and needs a human paraphrase edit. Near-zero score = candidate for the
"deliberately uses vocabulary absent from the source" criterion.

The >=0.50 threshold is triage, not an acceptance gate. It finds questions
worth a human look; the human decides. Do not rewrite a question purely to
move the number — the metric is crude (whole-file overlap, unstemmed, so
"arrows" scores and "shafts" does not; unavoidable proper nouns count
against you; short questions are penalised and long ones dilute), and
chasing it produces stilted questions that are no harder for BM25. A
question above 0.50 that a human has judged well-paraphrased is fine if the
reason is recorded in its notes.

Watch for the same defect arriving by a different route: a question that
restates its own answer smuggles the source's terms into the query even
when overall containment is low. That inflates BM25 specifically — sparse
matches literal terms, dense much less so — which corrupts the
dense-vs-sparse comparison the harness exists to make. Leaking the answer
to a *reader* in vocabulary absent from the source is not the same defect
and is not a problem for a retrieval-only MVP (see synthesis-05).

Deliberately not BM25 rank or the harness's own sparse retriever: iterating
against the retriever we're trying to evaluate would hand-engineer the sparse
arm to lose, corrupting the dense-vs-sparse comparison in the opposite
direction from what the spec is guarding against. Plain term overlap against
the passage text is decoupled from what's being measured.

Usage: python containment_check.py drafts.json
Reads a JSON array of question objects (id, question, stratum, source_document,
notes). Skips unanswerable questions (no source_document). Prints a table
sorted worst-first (highest containment = needs the most paraphrase work).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "lore_chunks"

# Small self-contained stopword list — no NLTK data download available/needed
# for a throwaway triage script.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "is", "was", "were", "are", "be", "been", "being", "am",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "he", "she", "it", "they", "them", "his", "her", "its", "their",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    "this", "that", "these", "those", "there", "here",
    "i", "you", "we", "me", "my", "your", "our", "us",
    "not", "no", "yes", "did", "does", "did", "can", "could", "would",
    "should", "will", "shall", "may", "might", "must", "s", "t",
}

WORD_RE = re.compile(r"[a-zA-Z]+")


def content_words(text: str) -> set[str]:
    return {
        w.lower()
        for w in WORD_RE.findall(text)
        if w.lower() not in STOPWORDS and len(w) > 2
    }


def containment(question_words: set[str], source_words: set[str]) -> float:
    if not question_words:
        return 0.0
    return len(question_words & source_words) / len(question_words)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} drafts.json", file=sys.stderr)
        return 1

    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data

    rows = []
    for q in questions:
        if q["stratum"] == "unanswerable" or not q.get("source_document"):
            continue
        source_path = CORPUS_DIR / q["source_document"]
        if not source_path.exists():
            rows.append((q["id"], -1.0, q["question"], set(), f"MISSING FILE: {source_path}"))
            continue
        source_text = source_path.read_text(encoding="utf-8")
        qwords = content_words(q["question"])
        swords = content_words(source_text)
        score = containment(qwords, swords)
        overlap = sorted(qwords & swords)
        rows.append((q["id"], score, q["question"], overlap, q["source_document"]))

    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"{'id':<16}{'score':>7}  question")
    print("-" * 100)
    for qid, score, question, overlap, source in rows:
        flag = " <-- MISSING SOURCE" if score < 0 else ""
        print(f"{qid:<16}{score:>7.2f}  {question}{flag}")
        print(f"{'':<16}{'':>7}  source: {source}")
        if overlap:
            print(f"{'':<16}{'':>7}  overlapping terms: {', '.join(overlap)}")
        print()

    high = [r for r in rows if r[1] >= 0.5]
    zero = [r for r in rows if 0 <= r[1] <= 0.05]
    print("=" * 100)
    print(f"Total in-corpus questions scored: {len(rows)}")
    print(f"High containment (>=0.5, needs paraphrase review): {len(high)} -> {[r[0] for r in high]}")
    print(f"Near-zero containment (<=0.05, candidates for 'absent vocabulary' criterion): {len(zero)} -> {[r[0] for r in zero]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
