from typing import List, Dict, Any
import warnings

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
        pairs = [(query, c['text']) for c in candidates]
        inputs = self.tokenizer([p[0] for p in pairs], [p[1] for p in pairs], return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1).cpu().numpy()
        for i, c in enumerate(candidates):
            c['rerank_score'] = float(scores[i])
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [candidates[i] for i in top_idx]
