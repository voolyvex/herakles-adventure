import logging
from typing import List, Dict, Any, Set
import os
import sys
import re
import warnings

# Add project root to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.name_mapping import get_name_variants, translate_name, GREEK_TO_ROMAN, ROMAN_TO_GREEK

# Configure logging
logger = logging.getLogger(__name__)
logger.propagate = False  # Don't propagate to root logger
if not logger.handlers:
    file_handler = logging.FileHandler('game.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

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
    """Agent for summarizing a list of texts into a concise context with name variant handling."""
    def __init__(self, model_name: str = 'facebook/bart-large-cnn'):
        try:
            self.summarizer = pipeline('summarization', model=model_name)
        except Exception as e:
            logger.warning(f"Could not load summarizer model: {e}")
            self.summarizer = None
    
    def _normalize_name_variants(self, text: str) -> str:
        """Normalize name variants in the text to a consistent form (prefer Greek names)."""
        if not text:
            return text
            
        # Split into words, preserving punctuation
        words = re.findall(r'\b\w+\b|\W+', text)
        result = []
        
        for word in words:
            # Skip non-word tokens (punctuation, etc.)
            if not word.strip() or not word.isalpha():
                result.append(word)
                continue
                
            # Check if this is a known name
            normalized, was_translated = translate_name(word, to_roman=False)
            if was_translated:
                result.append(normalized)
            else:
                result.append(word)
        
        return ''.join(result)

    def summarize(self, texts: List[str], max_length: int = 150, min_length: int = 30) -> str:
        """Summarize a list of texts, normalizing name variants in the result."""
        # Validate input
        if not texts:
            return "No information available."
            
        # Filter out empty texts
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            return "No valid information available."
        
        # Use fallback summarization if no transformer model available
        if not self.summarizer:
            return self._fallback_summarize(valid_texts, max_length)
        
        try:
            # Join texts with newlines to preserve structure
            joined = '\n'.join(valid_texts)
            # Clamp lengths - lower min_length to ensure summarizer runs more often
            safe_min_length = max(5, min(min_length, 64))
            safe_max_length = max(safe_min_length + 10, min(max_length, 512))
            # If input is too long, fallback to avoid excessive processing
            if len(joined.split()) > 1024:
                logger.debug(f"Input too long for summarization. Using fallback. Length: {len(joined.split())}")
                return self._fallback_summarize(valid_texts, max_length)
            # Generate summary
            summary_result = self.summarizer(
                joined, 
                max_length=safe_max_length, 
                min_length=safe_min_length,  # Clamp min_length
                do_sample=False,
                truncation=True
            )
            # Extract summary text with safer indexing
            if summary_result and len(summary_result) > 0 and 'summary_text' in summary_result[0]:
                summary = summary_result[0]['summary_text']
                # Normalize name variants in the summary
                normalized_summary = self._normalize_name_variants(summary)
                return normalized_summary
            else:
                # Fallback if summary result is unexpected
                logger.warning(f"Unexpected summary_result: {summary_result}")
                return self._fallback_summarize(valid_texts, max_length)
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            # Use fallback summarization method
            return self._fallback_summarize(valid_texts, max_length)
    
    def _fallback_summarize(self, texts: List[str], max_length: int = 150) -> str:
        """Simple fallback summarization method that doesn't use transformer models."""
        try:
            # If only one text, just truncate it
            if len(texts) == 1:
                return texts[0][:max_length] + ("..." if len(texts[0]) > max_length else "")
            
            # Extract first sentence from each text
            first_sentences = []
            for text in texts[:3]:  # Only use first 3 texts
                sentences = text.split('.')
                if sentences:
                    first_sentence = sentences[0].strip()
                    if first_sentence:
                        first_sentences.append(first_sentence)
            
            # Join first sentences
            if first_sentences:
                result = '. '.join(first_sentences)
                if len(result) > max_length:
                    result = result[:max_length] + "..."
                return result
            
            # If extraction failed, just concatenate and truncate
            concatenated = ' '.join(t[:50] for t in texts[:3])  # Take first 50 chars of first 3 texts
            return concatenated[:max_length] + ("..." if len(concatenated) > max_length else "")
            
        except Exception as e:
            logger.error(f"Error in fallback summarization: {e}")
            # Ultimate fallback
            return "Information is available but couldn't be summarized properly."
