from typing import List
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import with progress bars disabled
import torch
torch.set_num_threads(1)  # Limit CPU threads

# Import transformers with progress bars disabled
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from transformers import pipeline

class SummarizerAgent:
    """Agent for summarizing a list of texts into a concise context."""
    def __init__(self, model_name: str = 'facebook/bart-large-cnn'):
        self.summarizer = pipeline('summarization', model=model_name)

    def summarize(self, texts: List[str], max_length: int = 150) -> str:
        joined = '\n'.join(texts)
        summary = self.summarizer(joined, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']
