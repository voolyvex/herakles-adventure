"""Throwaway verification for ticket #8's unanswerable stratum.

Acceptance criterion: "Every unanswerable question is verified absent from
the corpus by querying the built index, not assumed absent." Grep over the
markdown source only catches lexical absence. This queries the real dense
retriever against the pinned index and prints the top-k so a human can judge
whether any returned passage actually *answers* the question — a name-drop
or thematic near-neighbor does not disqualify a question, only a passage
that would answer it does.

Mirrors myth_eval/cli.py's _build_real_retrievers wiring for the dense arm
only (sparse/hybrid/reranker are not needed to check absence).

Usage: python verify_unanswerable.py drafts.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# This script lives two directories below the repository root. Put that root on
# the import path so myth_eval and rag_system are importable no matter which
# directory the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} drafts.json", file=sys.stderr)
        return 1

    from myth_eval.index import DEFAULT_EMBEDDING_MODEL, index_root
    from myth_eval.retrieval import DenseRetrieverAdapter

    store = index_root()

    from rag_system import RAGSystem

    system = RAGSystem(
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        store_dir=str(store),
    )

    print(f"Index in use: {store}", file=sys.stderr)

    dense = DenseRetrieverAdapter(system.agentic_rag.dense)

    questions = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    unanswerable = [q for q in questions if q["stratum"] == "unanswerable"]

    results = []
    for q in unanswerable:
        hits = dense.retrieve(q["question"], k=5)
        row = {
            "id": q["id"],
            "question": q["question"],
            "top5": [
                {"source_document": h.source_document, "score": round(h.score, 4)}
                for h in hits
            ],
        }
        results.append(row)
        print(f"\n{q['id']}: {q['question']}")
        if not hits:
            print("  (no results)")
        for h in hits:
            print(f"  score={h.score:.4f}  {h.source_document}")

    out_path = Path(sys.argv[1]).with_name("unanswerable_index_check.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
