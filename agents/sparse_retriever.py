from typing import List, Dict, Any, Set, Optional
import os
import sys
import re
from rank_bm25 import BM25Okapi
import nltk

# Add project root to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.name_mapping import get_name_variants, translate_name, GREEK_TO_ROMAN, ROMAN_TO_GREEK

# Ensure 'punkt' is available for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SparseRetrieverAgent:
    """Agent for sparse keyword-based retrieval using BM25."""
    def __init__(self, lore_chunks: List[Dict[str, Any]]):
        self.lore_chunks = lore_chunks
        # Pre-tokenize the entire corpus once at initialization for efficiency.
        self.tokenized_corpus = [self._tokenize(chunk['text']) for chunk in lore_chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """A simple tokenizer to split text into words."""
        # This can be expanded with more sophisticated logic (e.g., stemming) if needed.
        return nltk.word_tokenize(text.lower())

    def retrieve(self, query_terms: List[str], k: int = 5, where_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieves lore chunks based on sparse keyword matching using BM25.
        """
        # If a filter is provided, we need to build a temporary index for the filtered chunks.
        if where_filter and 'god' in where_filter:
            god_variants = where_filter['god'].get('$in') or [where_filter['god'].get('$eq')]
            mask = [chunk.get('god', 'unknown') in god_variants for chunk in self.lore_chunks]
            
            # Apply the mask to get the filtered set of chunks and their tokenized versions
            filtered_chunks = [chunk for chunk, m in zip(self.lore_chunks, mask) if m]
            filtered_tokenized_corpus = [doc for doc, m in zip(self.tokenized_corpus, mask) if m]

            if not filtered_chunks:
                # If filter yields nothing, fall back to searching the entire corpus.
                bm25_index = self.bm25
                target_chunks = self.lore_chunks
            else:
                # Build a temporary BM25 index on the filtered corpus.
                bm25_index = BM25Okapi(filtered_tokenized_corpus)
                target_chunks = filtered_chunks
        else:
            # If no filter, use the main pre-built index.
            bm25_index = self.bm25
            target_chunks = self.lore_chunks
            
        # Score documents using the appropriate BM25 index.
        scores = bm25_index.get_scores(query_terms)
        
        # Get top k results.
        # Over-select internally to improve quality when k is large; then trim
        overselect = min(len(scores), max(k * 3, 50))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:overselect]
        
        # Return results with scores and debugging info.
        results = [
            {
                **target_chunks[i], 
                'score': float(scores[i]),
                'expanded_terms': query_terms
            } 
            for i in top_idx
        ]
        # Final trim to requested k
        return results[:k]
