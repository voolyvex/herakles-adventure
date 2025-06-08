from typing import List, Dict
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

class EntityAgent:
    """Agent for extracting named entities from user queries."""
    def __init__(self, model_name: str = 'dbmdz/bert-large-cased-finetuned-conll03-english'):
        self.ner = pipeline('ner', model=model_name, aggregation_strategy='simple')

    def extract_entities(self, query: str) -> List[Dict]:
        return self.ner(query)
