"""The seam test: a RAGSystem writes its store where the caller says.

This is the payoff for the expand–contract sequence. Before the store location
became a constructor parameter, no test could touch ``RAGSystem`` at all —
constructing one wrote into the developer's real ``chroma_db``. That is why the
production retrieval system had no tests, and why the rest of the suite's only
mention of it is an assertion that it stays unimported.

Everything here is marked ``heavy``: it loads a sentence-transformers model and
opens a Chroma store. Deselect with ``-m "not heavy"``. The heavy imports live
inside the test bodies, never at module level, because several other tests
assert that torch and chromadb are absent from ``sys.modules`` — a module-level
import here would fail them by collection order alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from myth_eval.index import repository_root

pytestmark = pytest.mark.heavy


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A two-document corpus, so indexing costs a moment rather than minutes."""
    directory = tmp_path / "lore"
    directory.mkdir()
    (directory / "001_apollo.md").write_text(
        "Apollo pursued Daphne, who became a laurel tree.", encoding="utf-8"
    )
    (directory / "002_herakles.md").write_text(
        "Herakles completed twelve labours for Eurystheus.", encoding="utf-8"
    )
    return directory


@pytest.fixture
def system(tmp_path: Path, corpus: Path):
    """A RAGSystem whose store is a temporary directory, not the real one."""
    from rag_system import RAGSystem

    return RAGSystem(
        lore_chunks_dir=str(corpus),
        store_dir=str(tmp_path / "store"),
        force_reindex=True,
    )


class TestTheStoreLandsWhereTheCallerSaid:
    def test_the_vector_store_is_written_to_the_stated_directory(
        self, system, tmp_path: Path
    ):
        store = tmp_path / "store"

        assert store.is_dir()
        assert any(store.iterdir()), "the stated directory holds no store"

    def test_the_chunk_cache_lands_beside_the_vector_store(self, system, tmp_path: Path):
        """The two relative literals had to move together: a store in one
        directory and its chunk cache in another is worse than neither."""
        assert (tmp_path / "store" / "lore_cache.json").is_file()

    def test_the_resolved_store_path_is_the_one_the_caller_stated(
        self, system, tmp_path: Path
    ):
        assert system.store_dir == (tmp_path / "store").resolve()

    def test_the_cache_accessor_reads_from_the_stated_directory(
        self, system, tmp_path: Path
    ):
        assert system._get_lore_cache_path().resolve() == (
            tmp_path / "store" / "lore_cache.json"
        ).resolve()

    def test_the_developers_real_store_is_left_untouched(self, system):
        """The reason this test could not be written before: construction used
        to write into whatever ``chroma_db`` the working directory named."""
        assert system.store_dir != (repository_root() / "chroma_db").resolve()

    def test_the_indexed_corpus_is_the_temporary_one(self, system):
        assert system.lore_chunks, "no chunks were indexed"
        sources = {chunk.get("source_file") for chunk in system.lore_chunks}
        assert sources <= {"001_apollo.md", "002_herakles.md"}, (
            f"indexed something outside the temporary corpus: {sources}"
        )


class TestAnAbsolutePathOutsideTheRepositoryWorks:
    def test_a_store_outside_the_repository_is_honoured(self, tmp_path: Path, corpus: Path):
        from rag_system import RAGSystem

        outside = tmp_path / "elsewhere" / "store"
        system = RAGSystem(
            lore_chunks_dir=str(corpus),
            store_dir=str(outside),
            force_reindex=True,
        )

        assert system.store_dir == outside.resolve()
        assert (outside / "lore_cache.json").is_file()
        assert repository_root() not in outside.resolve().parents
