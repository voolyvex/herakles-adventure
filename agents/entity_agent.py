"""Entity extraction agent using SpaCy for lightweight NER.

SpaCy's en_core_web_sm model is ~12MB (vs 340MB for BERT) and much faster.
Includes custom entity rules for Greek/Roman gods.
"""
from typing import List, Dict, Any, Set
import os
import sys
import logging

# Add project root to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.name_mapping import get_name_variants, GREEK_TO_ROMAN, ROMAN_TO_GREEK

import spacy
from spacy.language import Language


class EntityAgent:
    """Agent for extracting named entities using SpaCy with custom god name rules."""
    
    # All known god names (for custom entity ruler)
    GOD_NAMES: Set[str] = (
        set(GREEK_TO_ROMAN.keys()) | 
        set(ROMAN_TO_GREEK.keys()) |
        {v for v in GREEK_TO_ROMAN.values()} |
        {v for v in ROMAN_TO_GREEK.values()}
    )
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize SpaCy NER with custom god name patterns.
        
        Args:
            model_name: SpaCy model to load. Default is en_core_web_sm (~12MB).
                        The model_name parameter is kept for backward compatibility
                        but the BERT model name will be ignored.
        """
        try:
            # Map old BERT model names to SpaCy
            if "bert" in model_name.lower() or "dbmdz" in model_name.lower():
                logging.info(f"Ignoring BERT model '{model_name}', using SpaCy instead")
                model_name = "en_core_web_sm"
            
            self.nlp = spacy.load(model_name)
            self._add_god_entity_ruler()
            logging.info(f"EntityAgent initialized with SpaCy model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load SpaCy model: {e}")
            self.nlp = None
    
    def _add_god_entity_ruler(self) -> None:
        """Add custom entity patterns for Greek/Roman gods."""
        if self.nlp is None:
            return
            
        # Check if entity_ruler already exists
        if "entity_ruler" in self.nlp.pipe_names:
            return
            
        # Create patterns for all god names
        patterns = [
            {"label": "GOD", "pattern": name.capitalize()}
            for name in self.GOD_NAMES
        ]
        
        # Add the entity ruler before NER to prioritize god names
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(patterns)
    
    def _is_god_name(self, name: str) -> bool:
        """Check if a name is a known Greek or Roman god name."""
        return name.lower() in {n.lower() for n in self.GOD_NAMES}
    
    def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from query with special handling for god names.
        
        Args:
            query: Text to extract entities from.
            
        Returns:
            List of entity dictionaries with text, type, score, and position info.
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(query)
        
        processed_entities = []
        seen_entities: Set[str] = set()
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            entity_lower = entity_text.lower()
            
            # Skip duplicates
            if entity_lower in seen_entities:
                continue
            
            # Determine entity type
            if ent.label_ == "GOD" or self._is_god_name(entity_text):
                entity_type = "GOD"
                
                # Add the main entity
                processed_entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "score": 1.0,  # SpaCy doesn't provide scores, use 1.0
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
                seen_entities.add(entity_lower)
                
                # Add name variants
                variants = get_name_variants(entity_text)
                for variant in variants:
                    variant_lower = variant.lower()
                    if variant_lower != entity_lower and variant_lower not in seen_entities:
                        processed_entities.append({
                            "text": variant,
                            "type": "GOD_VARIANT",
                            "score": 0.9,
                            "source_entity": entity_text,
                        })
                        seen_entities.add(variant_lower)
            else:
                # Non-god entities
                processed_entities.append({
                    "text": entity_text,
                    "type": ent.label_,
                    "score": 1.0,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
                seen_entities.add(entity_lower)
        
        return processed_entities
