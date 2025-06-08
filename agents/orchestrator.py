from typing import List, Dict, Any, Optional
from .dense_retriever import DenseRetrieverAgent
from .sparse_retriever import SparseRetrieverAgent
from .reranker import RerankerAgent
from .summarizer import SummarizerAgent
from .entity_agent import EntityAgent

def deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for r in results:
        key = r['text']
        if key not in seen:
            deduped.append(r)
            seen.add(key)
    return deduped

class HybridOrchestrator:
    """Coordinates a team of retrieval agents for robust hybrid RAG."""
    def __init__(
        self,
        dense_agent: DenseRetrieverAgent,
        sparse_agent: SparseRetrieverAgent,
        reranker_agent: Optional[RerankerAgent] = None,
        summarizer_agent: Optional[SummarizerAgent] = None,
        entity_agent: Optional[EntityAgent] = None
    ):
        self.dense = dense_agent
        self.sparse = sparse_agent
        self.reranker = reranker_agent
        self.summarizer = summarizer_agent
        self.entity_agent = entity_agent

    def retrieve(self, query: str, k: int = 5) -> Dict[str, Any]:
        # 1. Optionally extract entities for focused retrieval
        if self.entity_agent:
            entities = self.entity_agent.extract_entities(query)
            # You could use entities to boost or filter retrieval here
        # 2. Parallel retrieval
        dense_hits = self.dense.retrieve(query, k=15)
        sparse_hits = self.sparse.retrieve(query, k=15)
        candidates = deduplicate(dense_hits + sparse_hits)
        # 3. Rerank
        if self.reranker:
            reranked = self.reranker.rerank(query, candidates, top_k=k)
        else:
            reranked = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)[:k]
        # 4. Summarize
        if self.summarizer:
            summary = self.summarizer.summarize([r['text'] for r in reranked])
        else:
            summary = '\n'.join([r['text'] for r in reranked])
        return {
            'summary': summary,
            'documents': reranked
        }
