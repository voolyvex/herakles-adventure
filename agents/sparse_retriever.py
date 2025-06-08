from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import nltk

# Ensure 'punkt' is available for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SparseRetrieverAgent:
    """Agent for sparse keyword-based retrieval using BM25."""
    def __init__(self, lore_chunks: List[Dict[str, Any]]):
        self.lore_chunks = lore_chunks
        self.corpus = [chunk['text'] for chunk in lore_chunks]
        self.tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.lore_chunks[i], 'score': float(scores[i])} for i in top_idx]
