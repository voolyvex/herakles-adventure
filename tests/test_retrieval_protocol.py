"""Tests for the retrieval protocol and its normalising adapters.

These use hand-built fake agents that return the exact hit shapes the real
dense and sparse agents produce. No ChromaDB, no embedding model, no network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from myth_eval.retrieval import (
    DenseRetrieverAdapter,
    RetrievedItem,
    Retriever,
    SparseRetrieverAdapter,
    normalise_hit,
)


class FakeDenseAgent:
    """Returns the shape DenseRetrieverAgent.retrieve returns.

    Tidy records, with source_file nested inside metadata.
    """

    def __init__(self, hits: Optional[List[Dict[str, Any]]] = None) -> None:
        self.hits = hits if hits is not None else _dense_hits()
        self.calls: List[tuple] = []

    def retrieve(self, query: str, k: int = 5, where_filter=None):
        self.calls.append((query, k, where_filter))
        return self.hits[:k]


class FakeSparseAgent:
    """Returns the shape SparseRetrieverAgent.retrieve returns.

    The whole raw chunk splatted out: source_file, god and title at the top
    level, metadata a sibling key. Accepts a pre-tokenised term list.
    """

    def __init__(self, hits: Optional[List[Dict[str, Any]]] = None) -> None:
        self.hits = hits if hits is not None else _sparse_hits()
        self.calls: List[tuple] = []

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().replace("?", "").split()

    def retrieve(self, query_terms: List[str], k: int = 5, where_filter=None):
        self.calls.append((query_terms, k, where_filter))
        return self.hits[:k]


def _dense_hits() -> List[Dict[str, Any]]:
    return [
        {
            "id": "005_APOLLO_AND_DAPHNE_c0",
            "text": "Apollo pursued Daphne through the woods.",
            "metadata": {
                "title": "APOLLO AND DAPHNE",
                "source_file": "005_APOLLO_AND_DAPHNE.md",
                "order": 0,
                "start_char": 0,
            },
            "score": 0.87,
            "query_used": "who chased Daphne",
        },
        {
            "id": "017_HYACINTHUS_c1",
            "text": "Hyacinthus was struck by the discus.",
            "metadata": {
                "title": "HYACINTHUS",
                "source_file": "017_HYACINTHUS.md",
                "order": 1,
                "start_char": 1200,
            },
            "score": 0.42,
            "query_used": "who chased Daphne",
        },
    ]


def _sparse_hits() -> List[Dict[str, Any]]:
    return [
        {
            "id": "005_APOLLO_AND_DAPHNE_c0",
            "text": "Apollo pursued Daphne through the woods.",
            "metadata": {
                "title": "APOLLO AND DAPHNE",
                "source_file": "005_APOLLO_AND_DAPHNE.md",
                "order": 0,
                "start_char": 0,
            },
            "god": "apollo",
            "god_variants": ["apollo"],
            "source_file": "005_APOLLO_AND_DAPHNE.md",
            "title": "APOLLO AND DAPHNE",
            "order": 0,
            "score": 14.203,
            "expanded_terms": ["who", "chased", "daphne"],
        },
        {
            "id": "012_PHAETON_c3",
            "text": "Phaeton drove the chariot of the sun.",
            "metadata": {
                "title": "PHAETON",
                "source_file": "012_PHAETON.md",
                "order": 3,
                "start_char": 3600,
            },
            "god": "unknown",
            "god_variants": ["apollo", "jupiter"],
            "source_file": "012_PHAETON.md",
            "title": "PHAETON",
            "order": 3,
            "score": 2.8,
            "expanded_terms": ["who", "chased", "daphne"],
        },
    ]


class TestProtocolConformance:
    def test_both_adapters_satisfy_the_retriever_protocol(self):
        dense = DenseRetrieverAdapter(FakeDenseAgent())
        sparse = SparseRetrieverAdapter(FakeSparseAgent())

        assert isinstance(dense, Retriever)
        assert isinstance(sparse, Retriever)

    def test_both_arms_are_reachable_through_one_call_shape(self):
        """The point of the protocol: identical call, no branching on arm."""
        arms = [
            DenseRetrieverAdapter(FakeDenseAgent()),
            SparseRetrieverAdapter(FakeSparseAgent()),
        ]

        for arm in arms:
            results = arm.retrieve("who chased Daphne?", k=2)
            assert all(isinstance(item, RetrievedItem) for item in results)

    def test_arms_carry_distinct_names(self):
        assert DenseRetrieverAdapter(FakeDenseAgent()).name == "dense"
        assert SparseRetrieverAdapter(FakeSparseAgent()).name == "sparse"


class TestUniformShape:
    def test_dense_and_sparse_produce_the_same_type_for_the_same_chunk(self):
        """The same underlying chunk normalises identically from either arm.

        Scores differ by construction (bounded cosine vs unbounded BM25); the
        identity fields must not.
        """
        dense = DenseRetrieverAdapter(FakeDenseAgent()).retrieve("q", k=1)[0]
        sparse = SparseRetrieverAdapter(FakeSparseAgent()).retrieve("q", k=1)[0]

        assert dense.chunk_id == sparse.chunk_id
        assert dense.source_document == sparse.source_document
        assert dense.text == sparse.text
        assert dense.rank == sparse.rank == 0

    def test_source_document_is_extracted_from_nested_metadata(self):
        """Dense nests source_file under metadata."""
        results = DenseRetrieverAdapter(FakeDenseAgent()).retrieve("q", k=2)

        assert results[0].source_document == "005_APOLLO_AND_DAPHNE.md"
        assert results[1].source_document == "017_HYACINTHUS.md"

    def test_source_document_is_extracted_from_top_level(self):
        """Sparse splats source_file at the top level."""
        results = SparseRetrieverAdapter(FakeSparseAgent()).retrieve("q", k=2)

        assert results[0].source_document == "005_APOLLO_AND_DAPHNE.md"
        assert results[1].source_document == "012_PHAETON.md"

    def test_ranks_are_dense_and_zero_based(self):
        results = DenseRetrieverAdapter(FakeDenseAgent()).retrieve("q", k=2)

        assert [item.rank for item in results] == [0, 1]

    def test_metadata_is_flattened_into_one_mapping(self):
        """Sparse's top-level god field and nested title both land in metadata."""
        item = SparseRetrieverAdapter(FakeSparseAgent()).retrieve("q", k=1)[0]

        assert item.metadata["god"] == "apollo"
        assert item.metadata["title"] == "APOLLO AND DAPHNE"

    def test_retrieved_items_are_immutable(self):
        """Metric code must not be able to mutate what it scores."""
        item = DenseRetrieverAdapter(FakeDenseAgent()).retrieve("q", k=1)[0]

        with pytest.raises(Exception):
            item.score = 0.0


class TestTokenisationIsHandledInsideTheAdapter:
    def test_sparse_adapter_accepts_a_string_and_passes_terms_to_the_agent(self):
        """The caller never tokenises; that difference is the adapter's job."""
        agent = FakeSparseAgent()

        SparseRetrieverAdapter(agent).retrieve("Who chased Daphne?", k=2)

        query_terms, k, _ = agent.calls[0]
        assert isinstance(query_terms, list)
        assert query_terms == ["who", "chased", "daphne"]

    def test_dense_adapter_passes_the_query_string_through_unchanged(self):
        agent = FakeDenseAgent()

        DenseRetrieverAdapter(agent).retrieve("Who chased Daphne?", k=2)

        query, _, _ = agent.calls[0]
        assert query == "Who chased Daphne?"

    def test_sparse_adapter_falls_back_when_agent_exposes_no_tokeniser(self):
        class TokeniserlessAgent:
            def retrieve(self, query_terms, k=5, where_filter=None):
                assert isinstance(query_terms, list)
                return []

        results = SparseRetrieverAdapter(TokeniserlessAgent()).retrieve("A B", k=3)

        assert results == []


class TestFiltersAndK:
    def test_filters_are_forwarded_to_both_agents(self):
        dense_agent, sparse_agent = FakeDenseAgent(), FakeSparseAgent()
        god_filter = {"god": {"$in": ["apollo"]}}

        DenseRetrieverAdapter(dense_agent).retrieve("q", k=2, filters=god_filter)
        SparseRetrieverAdapter(sparse_agent).retrieve("q", k=2, filters=god_filter)

        assert dense_agent.calls[0][2] == god_filter
        assert sparse_agent.calls[0][2] == god_filter

    def test_results_are_truncated_to_k(self):
        arm = DenseRetrieverAdapter(FakeDenseAgent())

        assert len(arm.retrieve("q", k=1)) == 1

    def test_k_larger_than_available_results_is_not_an_error(self):
        arm = DenseRetrieverAdapter(FakeDenseAgent())

        assert len(arm.retrieve("q", k=50)) == 2

    def test_an_agent_returning_nothing_yields_an_empty_list(self):
        arm = DenseRetrieverAdapter(FakeDenseAgent(hits=[]))

        assert arm.retrieve("q", k=5) == []

    def test_an_agent_returning_none_yields_an_empty_list(self):
        class NoneAgent:
            def retrieve(self, query, k=5, where_filter=None):
                return None

        assert DenseRetrieverAdapter(NoneAgent()).retrieve("q", k=5) == []


class TestNormaliseHitEdgeCases:
    def test_a_hit_with_no_source_file_normalises_to_an_empty_document(self):
        item = normalise_hit({"id": "x", "text": "t", "score": 1.0}, rank=0)

        assert item.source_document == ""

    def test_a_non_numeric_score_degrades_to_zero_rather_than_raising(self):
        item = normalise_hit({"id": "x", "text": "t", "score": None}, rank=0)

        assert item.score == 0.0

    def test_a_missing_score_defaults_to_zero(self):
        item = normalise_hit({"id": "x", "text": "t"}, rank=0)

        assert item.score == 0.0

    def test_nested_metadata_wins_over_a_conflicting_top_level_key(self):
        hit = {
            "id": "x",
            "text": "t",
            "score": 1.0,
            "title": "top-level title",
            "metadata": {"title": "curated title"},
        }

        assert normalise_hit(hit, rank=0).metadata["title"] == "curated title"
