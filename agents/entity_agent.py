from typing import List, Dict, Set, Tuple, Any
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

from transformers import pipeline

class EntityAgent:
    """Agent for extracting named entities from user queries with support for Greek/Roman variants."""
    def __init__(self, model_name: str = 'dbmdz/bert-large-cased-finetuned-conll03-english'):
        try:
            self.ner = pipeline('ner', model=model_name, aggregation_strategy='simple')
        except Exception as e:
            print(f"Warning: Could not load NER model: {e}")
            self.ner = None
            
    def _is_god_name(self, name: str) -> bool:
        """Check if a name is a known Greek or Roman god name."""
        name_lower = name.lower()
        return (name_lower in GREEK_TO_ROMAN or 
                name_lower in ROMAN_TO_GREEK or
                name_lower in {v.lower() for v in GREEK_TO_ROMAN.values()} or
                name_lower in {v.lower() for v in ROMAN_TO_GREEK.values()})
    
    def _get_name_variants(self, name: str) -> List[str]:
        """Get all Greek and Roman variants of a name."""
        return get_name_variants(name)
    
    def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from query, with special handling for god names."""
        if not self.ner:
            return []
            
        # Get standard NER results
        entities = self.ner(query)
        
        # Process entities to handle god names
        processed_entities = []
        seen_entities = set()
        
        for entity in entities:
            entity_text = entity['word'].strip()
            entity_lower = entity_text.lower()
            
            # Skip if we've already processed this entity
            if entity_lower in seen_entities:
                continue
                
            # Check if this is a god name
            if self._is_god_name(entity_text):
                # Add the original entity
                processed_entities.append({
                    'text': entity_text,
                    'type': 'GOD',
                    'score': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })
                
                # Add variants
                variants = self._get_name_variants(entity_text)
                for variant in variants:
                    if variant.lower() != entity_lower and variant.lower() not in seen_entities:
                        processed_entities.append({
                            'text': variant,
                            'type': 'GOD_VARIANT',
                            'score': entity['score'] * 0.9,  # Slightly lower score for variants
                            'source_entity': entity_text
                        })
                        seen_entities.add(variant.lower())
                
                seen_entities.add(entity_lower)
            else:
                # Add non-god entities as-is
                processed_entities.append(entity)
        
        return processed_entities
