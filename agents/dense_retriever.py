from typing import List, Dict, Any
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import with progress bars disabled
import torch
torch.set_num_threads(1)  # Limit CPU threads

# Import sentence-transformers with progress bars disabled
import logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
from sentence_transformers import SentenceTransformer

class DenseRetrieverAgent:
    """Agent for dense embedding-based retrieval using SentenceTransformer."""
    def __init__(self, embedding_model_name: str, lore_chunks: List[Dict[str, Any]]):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.lore_chunks = lore_chunks
        self.embeddings = self._embed_chunks()

    def _embed_chunks(self) -> np.ndarray:
        texts = [chunk['text'] for chunk in self.lore_chunks]
        return np.array(self.embedding_model.encode(texts, show_progress_bar=False))

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.embedding_model.encode([query])[0]
        scores = np.dot(self.embeddings, query_emb)
        top_idx = np.argsort(scores)[::-1][:k]
        return [{**self.lore_chunks[i], 'score': float(scores[i])} for i in top_idx]
