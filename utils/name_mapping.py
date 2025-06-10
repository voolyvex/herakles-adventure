"""Utility for handling Greek and Roman god name mappings."""
from typing import Dict, Optional, Tuple

# Mapping of Greek to Roman god names
GREEK_TO_ROMAN: Dict[str, str] = {
    # Major Olympians
    'zeus': 'jupiter',
    'hera': 'juno',
    'poseidon': 'neptune',
    'demeter': 'ceres',
    'athena': 'minerva',
    'apollo': 'apollo',
    'artemis': 'diana',
    'ares': 'mars',
    'aphrodite': 'venus',
    'hermes': 'mercury',
    'hephaestus': 'vulcan',
    'hestia': 'vesta',
    'dionysus': 'bacchus',
    'hades': 'pluto',
    'persephone': 'proserpina',
    'eros': 'cupid',
    'hebe': 'juventas',
    'hercules': 'hercules',
    'asclepius': 'aesculapius',
    'pan': 'faunus',
    'nemesis': 'invidia',
    'nike': 'victoria',
    'selene': 'luna',
    'eos': 'aurora',
    'helios': 'sol',
    'aeolus': 'vulturnus',
    'iris': 'arcus',
    'morpheus': 'somnus',
    'thanatos': 'mors',
    'hypnos': 'somnus',
}

# Create reverse mapping (Roman to Greek)
ROMAN_TO_GREEK: Dict[str, str] = {v: k for k, v in GREEK_TO_ROMAN.items()}

def translate_name(name: str, to_roman: bool = True) -> Tuple[str, bool]:
    """
    Translate between Greek and Roman god names.
    
    Args:
        name: The name to translate
        to_roman: If True, translate to Roman; if False, translate to Greek
        
    Returns:
        Tuple of (translated_name, was_translated)
    """
    if not name:
        return name, False
        
    name_lower = name.lower()
    
    if to_roman:
        # Greek to Roman
        if name_lower in GREEK_TO_ROMAN:
            return GREEK_TO_ROMAN[name_lower].capitalize(), True
        # Check if already Roman (case-insensitive)
        elif any(roman.lower() == name_lower for roman in ROMAN_TO_GREEK):
            return name, False
    else:
        # Roman to Greek
        if name_lower in ROMAN_TO_GREEK:
            return ROMAN_TO_GREEK[name_lower].capitalize(), True
        # Check if already Greek (case-insensitive)
        elif any(greek.lower() == name_lower for greek in GREEK_TO_ROMAN):
            return name, False
    
    # No translation found and not in either set
    return name, False

def normalize_god_name(name: str, target_style: str = 'roman') -> str:
    """
    Normalize a god name to either Greek or Roman form.
    
    Args:
        name: The name to normalize
        target_style: Either 'roman' or 'greek'
        
    Returns:
        The normalized name in the target style
    """
    if target_style.lower() == 'roman':
        normalized, _ = translate_name(name, to_roman=True)
    else:  # 'greek'
        normalized, _ = translate_name(name, to_roman=False)
    return normalized

def get_name_variants(name: str) -> list[str]:
    """
    Get all name variants (Greek and Roman) for a given god name.
    
    Args:
        name: The name to get variants for
        
    Returns:
        List of name variants (original, Greek, and Roman forms)
    """
    if not name:
        return [name] if name else []
        
    try:
        variants = {name.lower()}
        
        # Try to get both Greek and Roman variants
        roman, is_roman = translate_name(name, to_roman=True)
        if is_roman:
            variants.add(roman.lower())
        
        greek, is_greek = translate_name(name, to_roman=False)
        if is_greek:
            variants.add(greek.lower())
        
        # Add title-cased versions
        variants.update({v.capitalize() for v in variants})
        
        return list(variants)
    except Exception as e:
        # Fallback to just the original name in case of any error
        return [name]
