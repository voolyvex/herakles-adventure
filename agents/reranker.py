"""FlashRank-based reranker agent for fast CPU-optimized reranking.

FlashRank uses ONNX runtime and is 10x faster than transformer-based cross-encoders.
Models available:
- ms-marco-TinyBERT-L-2-v2 (~4MB): Blazing fast, competitive quality
- ms-marco-MiniLM-L-12-v2 (~34MB): Best quality, still very fast
- rank-T5-flan (~110MB): Best zero-shot performance
"""
from typing import List, Dict, Any
import logging

from flashrank import Ranker, RerankRequest


class RerankerAgent:
    """Agent for reranking retrieved passages using FlashRank (ONNX-optimized).
    
    FlashRank provides 10x faster reranking compared to transformer-based
    cross-encoders, with competitive quality on MS MARCO benchmarks.
    """
    
    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        max_length: int = 256,
        cache_dir: str = None,
    ):
        """Initialize the FlashRank reranker.
        
        Args:
            model_name: FlashRank model to use. Options:
                - "ms-marco-TinyBERT-L-2-v2" (~4MB, fastest)
                - "ms-marco-MiniLM-L-12-v2" (~34MB, best quality)
                - "rank-T5-flan" (~110MB, best zero-shot)
            max_length: Maximum sequence length for reranking.
            cache_dir: Directory to cache downloaded models.
        """
        logging.info(f"Initializing FlashRank reranker with model: {model_name}")
        
        # Map old model names to FlashRank equivalents
        model_mapping = {
            "BAAI/bge-reranker-v2-m3": "ms-marco-MiniLM-L-12-v2",
            "BAAI/bge-reranker-base": "ms-marco-MiniLM-L-12-v2",
            "BAAI/bge-reranker-large": "ms-marco-MiniLM-L-12-v2",
        }
        
        if model_name in model_mapping:
            model_name = model_mapping[model_name]
            logging.info(f"Mapped to FlashRank model: {model_name}")
        
        self.model_name = model_name
        
        # Build kwargs, only include cache_dir if specified
        ranker_kwargs = {"model_name": model_name, "max_length": max_length}
        if cache_dir is not None:
            ranker_kwargs["cache_dir"] = cache_dir
        
        self.ranker = Ranker(**ranker_kwargs)
        logging.info("FlashRank reranker initialized successfully")

    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank candidate documents against a query.
        
        Args:
            query: The search query.
            candidates: List of candidate documents with 'text' field.
            top_k: Number of top results to return.
            
        Returns:
            Top-k candidates sorted by rerank score, with 'rerank_score' added.
        """
        if not candidates:
            return []
        
        # Convert to FlashRank passage format
        passages = [
            {"id": str(i), "text": c.get("text", ""), "meta": c.get("metadata", {})}
            for i, c in enumerate(candidates)
        ]
        
        # Create rerank request and get results
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)
        
        # Map results back to original candidates with scores
        reranked = []
        for result in results[:top_k]:
            idx = int(result["id"])
            candidate = candidates[idx].copy()
            candidate["rerank_score"] = float(result["score"])
            candidate["rerank_query_used"] = query
            reranked.append(candidate)
        
        return reranked
