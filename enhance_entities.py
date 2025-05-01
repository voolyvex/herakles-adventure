"""
Enhanced entity extraction and management for mythology RPG.

This script:
1. Extracts entities from lore chunks
2. Updates lore_chunk JSON files with extracted entities 
3. Creates individual entity files in subdirectories
4. Establishes relationships between entities

Usage:
    python enhance_entities.py --extract-entities
    python enhance_entities.py --create-entity-files
    python enhance_entities.py --update-relationships
"""

import os
import re
import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple, Optional

# Directory structure
CHUNKS_DIR = Path("lore_chunks")
LORE_INDICES_DIR = Path("lore_entities")  # existing JSON files for lore chunks
ENTITIES_DIR = Path("entities")  # new directory for individual entity files

# Create entity subdirectories
ENTITY_TYPES = ["characters", "places", "items", "groups", "concepts"]

def setup_directories():
    """Create necessary directory structure if it doesn't exist."""
    ENTITIES_DIR.mkdir(exist_ok=True)
    for entity_type in ENTITY_TYPES:
        (ENTITIES_DIR / entity_type).mkdir(exist_ok=True)

def extract_entities_from_chunk(md_path: Path) -> Dict[str, List[str]]:
    """
    Extract entities from a markdown chunk using existing extraction logic.
    Returns a dictionary with keys 'characters', 'places', 'items' and corresponding lists.
    """
    # Import entity lists from extract_lore_entities.py to maintain consistency
    from extract_lore_entities import (
        KNOWN_CHARACTERS, KNOWN_PLACES, KNOWN_ITEMS, 
        CHARACTER_PAT, PLACE_PAT, ITEM_PAT, clean_text
    )
    
    # Read the markdown file
    with md_path.open("r", encoding="utf-8") as f:
        text = f.read()
    
    # Clean text before extraction
    cleaned_text = clean_text(text)
    
    # Extract potential entities using regex
    potential_chars = CHARACTER_PAT.findall(cleaned_text)
    potential_places = [m for m in PLACE_PAT.findall(cleaned_text)]
    potential_items = ITEM_PAT.findall(cleaned_text)
    
    # Filter against known lists
    chars = set(char for char in potential_chars if char in KNOWN_CHARACTERS)
    
    known_places_lower = {p.lower() for p in KNOWN_PLACES}
    places = set(place for place in potential_places if place.lower() in known_places_lower)
    
    items = set(item.capitalize() for item in potential_items 
                if item.lower() in {i.lower() for i in KNOWN_ITEMS})
    
    # Build final entity dictionary
    entities = defaultdict(list)
    if chars:
        entities['characters'] = sorted(list(chars))
    if places:
        entities['places'] = sorted(list(places))
    if items:
        entities['items'] = sorted(list(items))
    
    return entities

def update_lore_index_with_entities(lore_index_path: Path, entities: Dict[str, List[str]]) -> None:
    """Update the lore index JSON file with extracted entities."""
    try:
        with lore_index_path.open("r", encoding="utf-8") as f:
            lore_data = json.load(f)
        
        # Update entities field
        lore_data["entities"] = entities
        
        # Add a placeholder summary if not present
        if not lore_data.get("summary"):
            title = lore_data.get("title", "")
            lore_data["summary"] = f"Information about {title}"
        
        with lore_index_path.open("w", encoding="utf-8") as f:
            json.dump(lore_data, f, indent=2, ensure_ascii=False)
        
        print(f"Updated {lore_index_path.name} with {sum(len(v) for v in entities.values())} entities")
    
    except Exception as e:
        print(f"Error updating {lore_index_path.name}: {e}")

def create_entity_file(
    entity_name: str, 
    entity_type: str,
    lore_chunk_ids: List[str]
) -> None:
    """Create an individual entity file in the appropriate subdirectory."""
    # Normalize entity name for file system
    entity_id = entity_name.lower().replace(" ", "_")
    
    # Determine target directory based on entity type
    if entity_type == "character":
        target_dir = ENTITIES_DIR / "characters"
    elif entity_type == "place":
        target_dir = ENTITIES_DIR / "places"
    elif entity_type == "item":
        target_dir = ENTITIES_DIR / "items"
    elif entity_type == "group":
        target_dir = ENTITIES_DIR / "groups"
    elif entity_type == "concept":
        target_dir = ENTITIES_DIR / "concepts"
    else:
        print(f"Unknown entity type '{entity_type}' for {entity_name}, skipping")
        return
    
    # Ensure directory exists
    target_dir.mkdir(exist_ok=True)
    
    # Entity file path
    entity_path = target_dir / f"{entity_id}.json"
    
    # Create or update entity file
    if entity_path.exists():
        # Update existing entity file
        with entity_path.open("r", encoding="utf-8") as f:
            entity_data = json.load(f)
            
        # Update appearance_in_chunks (unique entries only)
        existing_chunks = set(entity_data.get("appearance_in_chunks", []))
        existing_chunks.update(lore_chunk_ids)
        entity_data["appearance_in_chunks"] = sorted(list(existing_chunks))
    else:
        # Create new entity file
        entity_data = {
            "id": entity_id,
            "name": entity_name,
            "type": entity_type.rstrip("s"),  # convert plural to singular
            "aliases": [],
            "description": f"{entity_name} - Extract more info from lore chunks",
            "appearance_in_chunks": sorted(lore_chunk_ids),
            "related_entities": [],
            "categories": []
        }
    
    # Write entity data
    with entity_path.open("w", encoding="utf-8") as f:
        json.dump(entity_data, f, indent=2, ensure_ascii=False)
    
    print(f"{'Updated' if entity_path.exists() else 'Created'} entity file for {entity_name}")

def extract_and_update_all_entities():
    """Process all lore chunks, extract entities, and update lore indices."""
    setup_directories()
    
    # Process each markdown file
    for md_path in CHUNKS_DIR.glob("*.md"):
        # Get corresponding JSON path
        json_path = LORE_INDICES_DIR / f"{md_path.stem}.json"
        
        if not json_path.exists():
            print(f"Warning: No JSON index for {md_path.name}, skipping")
            continue
        
        # Extract entities
        print(f"Processing {md_path.name}...")
        entities = extract_entities_from_chunk(md_path)
        
        # Update lore index
        update_lore_index_with_entities(json_path, entities)

def create_all_entity_files():
    """
    Create individual entity files for all entities referenced in lore indices.
    Determines entity types automatically based on the category in lore indices.
    """
    setup_directories()
    
    # Track all entities to avoid duplicates
    entity_references = defaultdict(list)  # name -> list of lore_chunk_ids
    
    # Scan all lore indices
    for json_path in LORE_INDICES_DIR.glob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                lore_data = json.load(f)
            
            lore_id = lore_data.get("id", "")
            entities = lore_data.get("entities", {})
            
            # Process each entity type
            for entity_type, entity_list in entities.items():
                for entity_name in entity_list:
                    # Store reference to this lore chunk
                    entity_references[(entity_name, entity_type)].append(lore_id)
        
        except Exception as e:
            print(f"Error processing {json_path.name}: {e}")
    
    # Create entity files
    for (entity_name, entity_type), lore_chunk_ids in entity_references.items():
        create_entity_file(entity_name, entity_type, lore_chunk_ids)
    
    print(f"Created/updated {len(entity_references)} entity files")

def update_entity_relationships():
    """
    Analyze entity appearances and create basic relationships.
    Co-occurrence in lore chunks suggests a relationship.
    """
    # First, build a mapping of which entities appear in which chunks
    chunk_entities = defaultdict(set)  # chunk_id -> set of entity ids
    entity_chunks = defaultdict(set)   # entity_id -> set of chunk_ids
    
    # Collect all entity files
    all_entity_files = []
    for entity_type in ENTITY_TYPES:
        all_entity_files.extend((ENTITIES_DIR / entity_type).glob("*.json"))
    
    # Build the mappings
    for entity_file in all_entity_files:
        try:
            with entity_file.open("r", encoding="utf-8") as f:
                entity_data = json.load(f)
            
            entity_id = entity_data.get("id", "")
            entity_chunks[entity_id] = set(entity_data.get("appearance_in_chunks", []))
            
            for chunk_id in entity_data.get("appearance_in_chunks", []):
                chunk_entities[chunk_id].add(entity_id)
        
        except Exception as e:
            print(f"Error processing entity file {entity_file.name}: {e}")
    
    # Now update relationships based on co-occurrence
    for entity_file in all_entity_files:
        try:
            with entity_file.open("r", encoding="utf-8") as f:
                entity_data = json.load(f)
            
            entity_id = entity_data.get("id", "")
            entity_type = entity_data.get("type", "")
            entity_name = entity_data.get("name", "")
            
            # Find co-occurring entities
            co_occurring_entities = set()
            for chunk_id in entity_chunks[entity_id]:
                co_occurring_entities.update(chunk_entities[chunk_id])
            
            # Remove self from co-occurring set
            if entity_id in co_occurring_entities:
                co_occurring_entities.remove(entity_id)
            
            # Convert set to list of relationship dictionaries
            relationships = []
            for co_entity_id in co_occurring_entities:
                # Find type of co-occurring entity
                co_entity_type = None
                for type_dir in ENTITY_TYPES:
                    if (ENTITIES_DIR / type_dir / f"{co_entity_id}.json").exists():
                        co_entity_type = type_dir
                        break
                
                if co_entity_type:
                    relationship = {
                        "id": co_entity_id,
                        "relationship": "associated_with",
                        "description": f"Appears in same lore as {entity_name}"
                    }
                    relationships.append(relationship)
            
            # Update entity file with relationships
            entity_data["related_entities"] = relationships
            
            with entity_file.open("w", encoding="utf-8") as f:
                json.dump(entity_data, f, indent=2, ensure_ascii=False)
            
            print(f"Updated relationships for {entity_name}: {len(relationships)} connections")
        
        except Exception as e:
            print(f"Error updating relationships for {entity_file.name}: {e}")

def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced entity extraction and management")
    
    parser.add_argument(
        "--extract-entities",
        action="store_true",
        help="Extract entities from lore chunks and update lore indices",
    )
    parser.add_argument(
        "--create-entity-files",
        action="store_true",
        help="Create individual entity files for all entities in lore indices",
    )
    parser.add_argument(
        "--update-relationships",
        action="store_true",
        help="Update entity relationship data based on co-occurrence",
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.extract_entities:
        extract_and_update_all_entities()
    elif args.create_entity_files:
        create_all_entity_files()
    elif args.update_relationships:
        update_entity_relationships()
    else:
        print("No action specified. Use --help for available commands.")

if __name__ == "__main__":
    main()
