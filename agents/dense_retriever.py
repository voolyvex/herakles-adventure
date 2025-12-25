from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import warnings
import re
import os
import sys

# Add project root to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.name_mapping import get_name_variants

# Suppress warnings
warnings.filterwarnings('ignore')

# Import with progress bars disabled
import torch
torch.set_num_threads(1)

# Import sentence-transformers with progress bars disabled
import logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
from sentence_transformers import SentenceTransformer
import chromadb

class DenseRetrieverAgent:
    """Agent for dense embedding-based retrieval using a ChromaDB collection."""
    
    # Models that require task prefixes
    PREFIXED_MODELS = {"nomic-ai/nomic-embed-text-v1.5", "nomic-embed-text-v1.5"}
    
    def __init__(
        self,
        collection: chromadb.Collection,
        embedding_model: Optional[SentenceTransformer] = None,
        embedding_model_name: Optional[str] = None,
        use_task_prefix: bool = False,
    ):
        """Initialize the dense retriever agent.
        
        Args:
            collection: ChromaDB collection to query.
            embedding_model: Pre-loaded SentenceTransformer model (preferred).
            embedding_model_name: Model name to load if embedding_model not provided.
            use_task_prefix: Whether to add task prefixes (e.g., for nomic models).
        """
        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif embedding_model_name is not None:
            self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        else:
            raise ValueError("Must provide either embedding_model or embedding_model_name")
        self.collection = collection
        self._use_task_prefix = use_task_prefix

    def retrieve(self, query: str, k: int = 5, where_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieves lore chunks from ChromaDB based on dense embedding similarity.
        """
        if self.collection.count() == 0:
            return []

        # Add task prefix if required (e.g., for nomic models)
        query_text = "search_query: " + query if self._use_task_prefix else query
        
        # Get embedding for the query (normalize for cosine similarity)
        query_emb = self.embedding_model.encode([query_text], normalize_embeddings=True).tolist()

        # Query the collection with the provided filter
        # Note: ChromaDB doesn't accept empty dict, use None instead
        query_kwargs = {
            "query_embeddings": query_emb,
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter
        
        results = self.collection.query(**query_kwargs)
        
        # Format results to be consistent with other agents
        retrieved_docs = []
        if results and results.get('ids')[0]:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                # Convert cosine distance to a similarity score (0 to 1)
                score = 1 - distance 
                retrieved_docs.append({
                    "id": doc_id,
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": float(score),
                    "query_used": query
                })

        return retrieved_docs
