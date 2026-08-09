"""Tests for index location pinning and cache-key correctness.

The acceptance criterion that matters most here is negative: changing chunk size
or overlap must *not* be able to reuse a stale index. The existing mtime-based
cache cannot make that distinction, which is the bug being fixed.

These tests build no vector store and load no model — they exercise key
computation and path resolution only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from myth_eval.index import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    IndexKey,
    compute_index_key,
    corpus_hash,
    index_is_current,
    index_root,
    read_index_key,
    repository_root,
    write_index_key,
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "lore"
    directory.mkdir()
    (directory / "001_A.md").write_text("Apollo pursued Daphne.", encoding="utf-8")
    (directory / "002_B.md").write_text("Phaeton drove the chariot.", encoding="utf-8")
    return directory


def _key(corpus: Path, **overrides) -> IndexKey:
    params = {
        "chunk_size_chars": DEFAULT_CHUNK_SIZE,
        "chunk_overlap_chars": DEFAULT_CHUNK_OVERLAP,
        "corpus_directory": corpus,
    }
    params.update(overrides)
    return compute_index_key(**params)


class TestPinnedLocation:
    def test_the_index_path_is_absolute(self):
        assert index_root().is_absolute()

    def test_the_index_path_does_not_depend_on_the_working_directory(self, tmp_path):
        """The defect being fixed: RAGSystem uses a CWD-relative "chroma_db",
        so the store lands wherever the process happened to start."""
        original = Path.cwd()
        try:
            from_repo = index_root()
            os.chdir(tmp_path)
            from_elsewhere = index_root()
        finally:
            os.chdir(original)

        assert from_repo == from_elsewhere

    def test_the_default_index_lives_under_the_repository_root(self):
        assert index_root() == repository_root() / "chroma_db"

    def test_a_relative_override_resolves_against_the_repository_root(self, tmp_path):
        original = Path.cwd()
        try:
            os.chdir(tmp_path)
            resolved = index_root("custom_store")
        finally:
            os.chdir(original)

        assert resolved == repository_root() / "custom_store"

    def test_an_absolute_override_is_respected(self, tmp_path):
        assert index_root(str(tmp_path / "store")) == (tmp_path / "store").resolve()


class TestChunkerParametersAreInTheKey:
    """The core acceptance criterion of this ticket."""

    def test_changing_chunk_size_changes_the_key(self, corpus):
        smaller = _key(corpus, chunk_size_chars=600)
        larger = _key(corpus, chunk_size_chars=1200)

        assert smaller != larger
        assert smaller.fingerprint() != larger.fingerprint()

    def test_changing_chunk_overlap_changes_the_key(self, corpus):
        assert _key(corpus, chunk_overlap_chars=0) != _key(
            corpus, chunk_overlap_chars=200
        )

    def test_changing_chunk_size_invalidates_a_built_index(self, corpus, tmp_path):
        """Under the old mtime-only cache this reused the stale index silently:
        changing chunk size touches no corpus file."""
        store = tmp_path / "store"
        write_index_key(_key(corpus, chunk_size_chars=1200), store)

        assert index_is_current(_key(corpus, chunk_size_chars=1200), store)
        assert not index_is_current(_key(corpus, chunk_size_chars=600), store)

    def test_changing_overlap_invalidates_a_built_index(self, corpus, tmp_path):
        store = tmp_path / "store"
        write_index_key(_key(corpus, chunk_overlap_chars=200), store)

        assert not index_is_current(_key(corpus, chunk_overlap_chars=50), store)

    def test_changing_the_embedding_model_changes_the_key(self, corpus):
        """Re-embedding the same chunks with a different model is a different
        vector space — relevant to the pending bge-small to nomic upgrade."""
        assert _key(corpus, embedding_model="BAAI/bge-small-en-v1.5") != _key(
            corpus, embedding_model="nomic-ai/nomic-embed-text-v1.5"
        )


class TestCorpusHash:
    def test_the_hash_is_stable_across_repeated_calls(self, corpus):
        assert corpus_hash(corpus) == corpus_hash(corpus)

    def test_editing_a_corpus_file_changes_the_hash(self, corpus):
        before = corpus_hash(corpus)
        (corpus / "001_A.md").write_text("Apollo pursued Daphne, who fled.", "utf-8")

        assert corpus_hash(corpus) != before

    def test_adding_a_corpus_file_changes_the_hash(self, corpus):
        before = corpus_hash(corpus)
        (corpus / "003_C.md").write_text("Midas turned things to gold.", "utf-8")

        assert corpus_hash(corpus) != before

    def test_touching_a_file_without_editing_it_does_not_change_the_hash(self, corpus):
        """Content-hashed, not mtime-based: a fresh checkout rewrites every
        mtime and must not force a rebuild."""
        before = corpus_hash(corpus)
        os.utime(corpus / "001_A.md", (0, 0))

        assert corpus_hash(corpus) == before

    def test_a_missing_corpus_directory_hashes_without_raising(self, tmp_path):
        assert isinstance(corpus_hash(tmp_path / "absent"), str)


class TestKeyPersistence:
    def test_a_written_key_reads_back_identically(self, corpus, tmp_path):
        store = tmp_path / "store"
        key = _key(corpus)

        write_index_key(key, store)

        assert read_index_key(store) == key

    def test_reading_from_a_location_with_no_index_returns_none(self, tmp_path):
        assert read_index_key(tmp_path / "empty") is None

    def test_a_corrupt_key_file_reads_as_none_rather_than_raising(self, tmp_path):
        store = tmp_path / "store"
        store.mkdir()
        (store / "index_key.json").write_text("{not json", encoding="utf-8")

        assert read_index_key(store) is None

    def test_an_index_with_no_recorded_key_is_never_current(self, corpus, tmp_path):
        assert not index_is_current(_key(corpus), tmp_path / "empty")

    def test_the_written_key_records_its_fingerprint(self, corpus, tmp_path):
        store = tmp_path / "store"
        key = _key(corpus)
        write_index_key(key, store)

        import json

        written = json.loads((store / "index_key.json").read_text(encoding="utf-8"))

        assert written["fingerprint"] == key.fingerprint()
        assert written["chunk_size_chars"] == key.chunk_size_chars
        assert written["chunk_overlap_chars"] == key.chunk_overlap_chars


class TestKeyComputationIsCheap:
    def test_computing_a_key_loads_no_heavy_dependencies(self):
        """CI decides whether a cache hit is valid without loading a model,
        so key computation must not import torch or chromadb."""
        import myth_eval.index as index_module

        source = open(index_module.__file__).read()
        header = source.split("def build_index")[0]

        for forbidden in ("import torch", "import chromadb", "from rag_system"):
            assert forbidden not in header
