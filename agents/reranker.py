from typing import List, Dict, Any, Tuple, Set
import os
import sys
import re
import warnings

# Add project root to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.name_mapping import get_name_variants, translate_name, GREEK_TO_ROMAN, ROMAN_TO_GREEK

# Suppress warnings
warnings.filterwarnings('ignore')

# Import with progress bars disabled
import torch
torch.set_num_threads(1)  # Limit CPU threads

# Import transformers with progress bars disabled
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RerankerAgent:
    """Agent for reranking retrieved passages using a cross-encoder."""
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents against an original query.
        """
        # Create query-document pairs for reranking using the original query
        pairs = [(query, c['text']) for c in candidates]
        
        # Tokenize and get scores
        inputs = self.tokenizer(
            [p[0] for p in pairs], 
            [p[1] for p in pairs], 
            return_tensors='pt', 
            padding=True, 
            truncation='only_second',  # Truncate document, not query
            max_length=512  # Ensure we don't exceed model's max length
        )
        
        # Get scores from the reranker model
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        # Update candidates with rerank scores
        for i, c in enumerate(candidates):
            c['rerank_score'] = float(scores[i])
        
        # Sort by rerank score and return top k
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Add debug info to top candidates
        for i in top_idx[:3]:  # Only add to top candidates to avoid too much output
            candidates[i]['rerank_query_used'] = query
        
        return [candidates[i] for i in top_idx]
