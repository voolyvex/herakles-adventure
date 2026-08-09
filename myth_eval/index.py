"""Explicit index building at a stated location.

Two problems this fixes.

**The store must land in one predictable place.** :func:`index_root` resolves
the store directory against the repository root — honouring an explicit
override or ``MYTH_INDEX_DIR`` — and :func:`build_index` passes that resolved
path to ``RAGSystem``. Because the location is stated rather than inferred from
the working directory, running the app from the repository root and the
evaluation from anywhere else reach the same store, and CI cannot build one and
read another.

**A re-chunk silently reuses a stale index.** ``RAGSystem._is_cache_fresh``
compares the cache's mtime against corpus file mtimes only. Changing
``chunk_size_chars`` or ``chunk_overlap_chars`` touches no corpus file, so the
cache stays "fresh" and the old chunking is reused under the new settings.
:func:`compute_index_key` folds the chunker parameters into the key alongside a
content hash of the corpus, so changing chunk size or overlap forces a rebuild.

Building the index is also made an explicit, separately invocable step rather
than a side effect of constructing the RAG system:

    python -m myth_eval.index --force
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "IndexKey",
    "repository_root",
    "index_root",
    "corpus_hash",
    "compute_index_key",
    "read_index_key",
    "write_index_key",
    "index_is_current",
    "build_index",
    "main",
]

# Default chunker settings, mirroring RAGSystem's own defaults. Duplicated here
# rather than imported so that computing an index key stays a cheap, dependency
# -free operation: importing rag_system pulls in torch and chromadb.
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_COLLECTION = "myth_lore"
DEFAULT_CORPUS_DIR = "lore_chunks"

# Name of the file recording which key the built index corresponds to. Lives
# beside the store so a store and its provenance travel together.
KEY_FILENAME = "index_key.json"


@dataclass(frozen=True)
class IndexKey:
    """Everything that, if changed, invalidates a built index.

    Chunker parameters are members precisely because the existing mtime-based
    cache omits them, which is the bug this fixes. The embedding model is a
    member because re-embedding the same chunks with a different model produces
    a different vector space — relevant to the pending bge-small to
    nomic-embed-text-v1.5 upgrade this harness exists partly to adjudicate.
    """

    corpus_hash: str
    chunk_size_chars: int
    chunk_overlap_chars: int
    embedding_model: str
    collection_name: str

    def fingerprint(self) -> str:
        """A short stable digest of the whole key, for cache keys and paths."""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint()
        return data


def repository_root() -> Path:
    """The repository root, resolved from this file's location.

    Deliberately independent of the current working directory: every path this
    module resolves is anchored here so that none of them shift with it.
    """
    return Path(__file__).resolve().parent.parent


def index_root(override: Optional[str] = None) -> Path:
    """The pinned directory the vector store lives in.

    Args:
        override: An explicit path, or the ``MYTH_INDEX_DIR`` environment
            variable, or the default ``<repo>/chroma_db``. A relative override
            is resolved against the repository root, not the working directory.

    Returns:
        An absolute path.
    """
    raw = override or os.getenv("MYTH_INDEX_DIR") or "chroma_db"
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = repository_root() / candidate
    return candidate.resolve()


def corpus_dir(override: Optional[str] = None) -> Path:
    """The corpus directory, resolved against the repository root."""
    raw = override or os.getenv("MYTH_CORPUS_DIR") or DEFAULT_CORPUS_DIR
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = repository_root() / candidate
    return candidate.resolve()


def corpus_hash(directory: Optional[Path] = None) -> str:
    """A content hash over every corpus file.

    Hashes filenames and contents rather than modification times, so that a
    checkout — which rewrites mtimes wholesale — does not spuriously invalidate,
    and an edit that preserves mtime does not spuriously validate.
    """
    target = directory if directory is not None else corpus_dir()
    digest = hashlib.sha256()

    if not target.exists():
        logger.warning("Corpus directory not found: %s", target)
        return digest.hexdigest()

    for path in sorted(target.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())

    return digest.hexdigest()


def compute_index_key(
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    corpus_directory: Optional[Path] = None,
) -> IndexKey:
    """Build the key describing the index these settings would produce."""
    return IndexKey(
        corpus_hash=corpus_hash(corpus_directory),
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        embedding_model=embedding_model,
        collection_name=collection_name,
    )


def read_index_key(directory: Optional[Path] = None) -> Optional[IndexKey]:
    """Read the key of an already-built index, or None if there is none."""
    path = (directory if directory is not None else index_root()) / KEY_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return IndexKey(
            corpus_hash=data["corpus_hash"],
            chunk_size_chars=int(data["chunk_size_chars"]),
            chunk_overlap_chars=int(data["chunk_overlap_chars"]),
            embedding_model=data["embedding_model"],
            collection_name=data["collection_name"],
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.warning("Could not read index key at %s: %s", path, error)
        return None


def write_index_key(key: IndexKey, directory: Optional[Path] = None) -> Path:
    """Record the key an index was built under, beside the store."""
    target = directory if directory is not None else index_root()
    target.mkdir(parents=True, exist_ok=True)
    path = target / KEY_FILENAME
    path.write_text(json.dumps(key.to_dict(), indent=2, sort_keys=True), "utf-8")
    return path


def index_is_current(key: IndexKey, directory: Optional[Path] = None) -> bool:
    """Whether a built index matches ``key`` exactly.

    Any difference — corpus content, chunk size, overlap, embedding model —
    means the built index does not correspond to these settings and must be
    rebuilt. This is the check the mtime-based cache cannot make.
    """
    return read_index_key(directory) == key


def build_index(
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    force: bool = False,
    directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the vector store at the pinned location, as an explicit step.

    Skips the work when a current index already exists, unless ``force``.

    Importing and constructing ``RAGSystem`` is deferred into this function
    because it pulls in torch, chromadb and sentence-transformers. Everything
    above — key computation, staleness checks — stays importable in a minimal
    environment, which is what lets CI decide whether a cache hit is valid
    without loading a model.

    Returns:
        A summary dict: the key, the resolved store path, whether a rebuild
        happened, and the indexed chunk count.
    """
    target = index_root(directory)
    key = compute_index_key(
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        embedding_model=embedding_model,
        collection_name=collection_name,
    )

    if not force and index_is_current(key, target):
        logger.info("Index at %s is current (%s); skipping build", target, key.fingerprint())
        return {
            "index_key": key.to_dict(),
            "index_path": str(target),
            "rebuilt": False,
            "chunk_count": None,
        }

    target.mkdir(parents=True, exist_ok=True)

    stale_key_path = target / KEY_FILENAME
    if stale_key_path.exists():
        # Remove first: if the build fails part-way, a stale key must not be
        # left claiming the half-built index is current.
        stale_key_path.unlink()

    from rag_system import RAGSystem  # noqa: PLC0415 — deliberately deferred

    # The store location is stated, not inferred from the working directory.
    system = RAGSystem(
        lore_chunks_dir=str(corpus_dir()),
        embedding_model_name=embedding_model,
        collection_name=collection_name,
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        force_reindex=True,
        store_dir=str(target),
    )
    chunk_count = len(getattr(system, "lore_chunks", []) or [])

    write_index_key(key, target)
    logger.info("Built index at %s (%s), %d chunks", target, key.fingerprint(), chunk_count)

    return {
        "index_key": key.to_dict(),
        "index_path": str(target),
        "rebuilt": True,
        "chunk_count": chunk_count,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point: ``python -m myth_eval.index``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the lore vector store at a pinned location.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if the index is current."
    )
    parser.add_argument(
        "--print-key",
        action="store_true",
        help="Print the index key and exit without building. Cheap: loads no model.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.print_key:
        key = compute_index_key(
            chunk_size_chars=args.chunk_size,
            chunk_overlap_chars=args.chunk_overlap,
            embedding_model=args.embedding_model,
            collection_name=args.collection,
        )
        print(json.dumps(key.to_dict(), indent=2, sort_keys=True))
        return 0

    summary = build_index(
        chunk_size_chars=args.chunk_size,
        chunk_overlap_chars=args.chunk_overlap,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        force=args.force,
        directory=args.index_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
