from typing import List, Dict, Any, Tuple, Optional
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
    def __init__(self, embedding_model_name: str, collection: chromadb.Collection):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection = collection

    def retrieve(self, query: str, k: int = 5, where_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieves lore chunks from ChromaDB based on dense embedding similarity.
        """
        if self.collection.count() == 0:
            return []

        # Get embedding for the query
        query_emb = self.embedding_model.encode([query]).tolist()

        # Query the collection with the provided filter
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k,
            where=where_filter if where_filter else {},
            include=["documents", "metadatas", "distances"]
        )
        
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
