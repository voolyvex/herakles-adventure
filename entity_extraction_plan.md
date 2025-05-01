# Mythology RPG Entity Database Plan

## Revised Data Model

1. **Primary Structure:**
   - `lore_chunks/`: Primary source markdown files (existing)
   - `lore_entities/`: JSON index files for each chunk, type="lore_chunk" (existing, now fixed)
   - `entities/`: New directory for extracted entity files (to be created)
      - Organized by type: `characters/`, `places/`, `items/`, `groups/`, `concepts/`

2. **Data Flow:**
   - `lore_chunks/*.md` → Extract entities → Update `lore_entities/*.json` → Generate `entities/*/*.json`

## Implementation Steps

### Step 1: Extract Entities from Lore Chunks ✓
- We already have `extract_lore_entities.py` that identifies entities in chunks using dictionaries and regex
- It currently generates a simple list of entities by type

### Step 2: Enhanced Entity Extraction (Next Task)
- Build on existing extraction code
- For each markdown chunk:
  1. Process text to find characters, places, items
  2. Update corresponding lore index JSON with references to these entities
  3. Generate/update individual entity files for each extracted entity

### Step 3: Entity Data Collection
- For each unique entity:
  1. Create a file in the appropriate entity subdirectory
  2. Record all chunks that mention this entity
  3. Define relationships to other entities

### Step 4: Relationship Mapping
- Define hierarchies (e.g., Zeus is father of Apollo)
- Define associations (e.g., Thor owns Mjolnir)
- Track spatial relationships (e.g., Mount Olympus is home to the gods)

## Enhanced Entity Structure (Sample)

```json
// entities/characters/hercules.json
{
  "id": "hercules",
  "name": "Hercules",
  "type": "character",
  "aliases": ["Heracles", "Alcides"], 
  "description": "Son of Zeus and mortal woman Alcmene, famous for his twelve labors",
  "appearance_in_chunks": ["047_HERCULES", "048_THE_TWELVE_LABORS"],
  "related_entities": [
    {"id": "zeus", "relationship": "parent", "description": "Zeus is Hercules' father"},
    {"id": "hera", "relationship": "enemy", "description": "Hera persecuted Hercules throughout his life"}
  ],
  "categories": ["hero", "demigod", "greek"]
}
```

## Refactored Lore Chunk Structure (Sample) 

```json
// lore_entities/047_HERCULES.json
{
  "id": "047_HERCULES",
  "title": "Hercules",
  "type": "lore_chunk",
  "source_file": "047_HERCULES.md",
  "entities": {
    "characters": ["Hercules", "Zeus", "Hera", "Eurystheus"],
    "places": ["Mount Olympus", "Thebes", "Nemea"],
    "items": ["Nemean Lion Skin", "Club"]
  },
  "summary": "The tale of Hercules, his origins, and accomplishments",
  "notes": ""
}
```
