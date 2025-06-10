from typing import List, Dict, Any, Optional, Set
import os
import sys
import nltk
from nltk.tokenize import word_tokenize
import re
import time
from concurrent.futures import ThreadPoolExecutor

# Add project root to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.name_mapping import get_name_variants, translate_name, GREEK_TO_ROMAN, ROMAN_TO_GREEK

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
        entity_agent: Optional[EntityAgent] = None,
        summarizer_agent: Optional[SummarizerAgent] = None
    ):
        self.dense = dense_agent
        self.sparse = sparse_agent
        self.reranker = reranker_agent
        self.entity_agent = entity_agent
        self.summarizer = summarizer_agent
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def _expand_query_with_name_variants(self, query: str) -> str:
        """Expand query with Greek and Roman name variants."""
        words = query.split()
        expanded_query = []
        
        for word in words:
            # Clean word (remove punctuation, lowercase)
            clean_word = ''.join(c.lower() for c in word if c.isalpha())
            if not clean_word:
                continue
                
            # Get all name variants for this word
            variants = get_name_variants(clean_word)
            expanded_query.extend(variants)
        
        # Combine original query with variants, removing duplicates while preserving order
        all_terms = words + expanded_query
        seen = set()
        result = []
        for term in all_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                result.append(term)
        
        return ' '.join(result)

    def _extractive_summary(self, query: str, documents: List[Dict[str, Any]], num_sentences: int = 3) -> str:
        """
        Creates a simple extractive summary from the top documents.
        """
        if not documents:
            return "No relevant information found."

        # Combine text from top documents
        full_text = ". ".join([doc.get('text', '') for doc in documents[:2]]) # Use top 2 docs
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', full_text)
        
        query_terms = set(word_tokenize(query.lower()))

        # Score sentences based on overlap with query terms
        scored_sentences = []
        for sentence in sentences:
            if not sentence or len(sentence.split()) < 5: # Skip very short sentences
                continue
            
            sentence_terms = set(word_tokenize(sentence.lower()))
            score = len(query_terms.intersection(sentence_terms))
            scored_sentences.append((score, sentence))
            
        # Sort sentences by score and take the top ones
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        top_sentences = [s for score, s in scored_sentences if score > 0][:num_sentences]

        if not top_sentences:
            return (documents[0]['text'][:250] + '...') if len(documents[0]['text']) > 250 else documents[0]['text']

        return " ".join(top_sentences)

    def retrieve(self, query: str, k: int = 5, god_context: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Establish God Context Filter (Primary)
        god_filter = None
        if god_context:
            god_variants = get_name_variants(god_context)
            if god_variants:
                god_filter = {"god": {"$in": god_variants}}
                print(f"[RAG] Filtering for god: {god_context} (variants: {god_variants})")

        # 2. Expand query for broad retrieval
        expanded_query = self._expand_query_with_name_variants(query)
        expanded_terms = word_tokenize(expanded_query.lower())

        # 3. Parallel retrieval with the god filter
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self.dense.retrieve, expanded_query, 15, god_filter)
            sparse_future = executor.submit(self.sparse.retrieve, expanded_terms, 15, god_filter) if self.sparse else None
            dense_hits = dense_future.result()
            sparse_hits = sparse_future.result() if sparse_future else []
        retrieval_time = time.time() - start_time
        print(f"[RAG] Retrieval took {retrieval_time:.2f}s")
        
        candidates = deduplicate(dense_hits + sparse_hits)
        if not candidates:
            return {
                'summary': "I couldn't find any relevant information on that topic.",
                'documents': [],
                'top_docs': [],
                'ranked_docs': []
            }

        # 4. Rerank with original query for better precision
        rerank_start = time.time()
        # Rerank a smaller pool of candidates to improve performance
        candidates_to_rerank = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)[:10]
        try:
            if self.reranker:
                reranked = self.reranker.rerank(query, candidates_to_rerank, top_k=k)
            else:
                reranked = candidates_to_rerank[:k]
        except Exception as e:
            print(f"Warning: Error in reranking: {e}")
            reranked = candidates_to_rerank[:k]
        rerank_time = time.time() - rerank_start
        print(f"[RAG] Reranking took {rerank_time:.2f}s")
        
        if not reranked:
            reranked = candidates[:k]

        # 5. Create summary from top results
        summary_start = time.time()
        summary = "No relevant information found on this topic."
        docs_for_summary = reranked[:3]  # Use top 3 docs for summary

        if docs_for_summary and self.summarizer:
            try:
                texts_for_summary = [doc['text'] for doc in docs_for_summary if doc.get('text')]
                if texts_for_summary:
                    # Generate a concise summary of the most relevant info
                    summary = self.summarizer.summarize(
                        texts_for_summary, 
                        max_length=120, 
                        min_length=25
                    )
            except Exception as e:
                print(f"Warning: Error in summarization: {e}")
                # Fallback to simple concatenation if summarizer fails
                summary = ". ".join([doc['text'][:150] for doc in docs_for_summary if doc.get('text')])
        elif docs_for_summary:
            # Fallback if summarizer agent isn't present for some reason
            summary = ". ".join([doc['text'][:150] for doc in docs_for_summary if doc.get('text')])

        summary_time = time.time() - summary_start
        print(f"[RAG] Summarization took {summary_time:.2f}s")

        total_time = time.time() - start_time
        print(f"[RAG] Total RAG pipeline took {total_time:.2f}s")
        return {
            'summary': summary,
            'documents': reranked,
            'top_docs': reranked,
            'ranked_docs': reranked
        }
