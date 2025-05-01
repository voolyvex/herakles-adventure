# Proper Mythological RPG Data Model

## Directory Structure

```
myth-rpg/
├── lore_chunks/                # Primary source markdown files
│   ├── 001_INTRODUCTION.md
│   ├── 002_ROMAN_DIVINITIES.md
│   └── ...
├── lore_indices/              # JSON files that index/categorize the lore chunks
│   ├── 001_INTRODUCTION.json
│   ├── 002_ROMAN_DIVINITIES.json
│   └── ...
├── entities/                  # Individual entity files (extracted from chunks)
│   ├── characters/            # Individual beings (gods, heroes, creatures)
│   │   ├── zeus.json
│   │   ├── hercules.json
│   │   └── ...
│   ├── places/                # Physical locations
│   │   ├── mount_olympus.json
│   │   ├── underworld.json
│   │   └── ...
│   ├── items/                 # Physical objects/artifacts
│   │   ├── golden_fleece.json
│   │   ├── thors_hammer.json
│   │   └── ...
│   ├── groups/                # Collections of characters/organizations
│   │   ├── olympian_gods.json
│   │   ├── argonauts.json
│   │   └── ...
│   └── concepts/              # Abstract ideas/events
│       ├── ragnarok.json
│       ├── fate.json
│       └── ...
├── relationships/             # Optional directory for storing relationship data
│   └── relationship_graph.json
└── scripts/                   # Processing/extraction scripts
    ├── extract_entities.py    # Extract entities from lore chunks
    ├── build_indices.py       # Build indices for lore chunks
    └── ...
```

## File Structures

### Lore Index Files (lore_indices/*.json)

```json
{
  "id": "047_HERCULES",
  "title": "Hercules",
  "type": "lore_chunk",
  "source_file": "047_HERCULES.md",
  "referenced_entities": [
    "hercules",
    "zeus",
    "hera",
    "twelve_labors"
  ],
  "summary": "",
  "notes": ""
}
```

### Entity Files (entities/*/*.json)

```json
{
  "id": "hercules",
  "name": "Hercules",
  "type": "character",
  "aliases": ["Heracles", "Alcides"],
  "description": "Son of Zeus and mortal woman Alcmene, famous for his twelve labors",
  "appearance_in_chunks": [
    "047_HERCULES",
    "048_THE_TWELVE_LABORS"
  ],
  "related_entities": [
    {
      "id": "zeus",
      "relationship": "parent",
      "description": "Zeus is Hercules' father"
    },
    {
      "id": "hera",
      "relationship": "enemy",
      "description": "Hera persecuted Hercules throughout his life"
    }
  ],
  "categories": ["hero", "demigod", "greek"],
  "attributes": {
    "strength": "legendary",
    "origin": "greek"
  }
}
```

## Extraction Process

1. **Parse Lore Chunks**:
   - Process each markdown file
   - Identify entities mentioned (using NLP/regex/dictionary approaches)
   - Create a lore index file that references these entities

2. **Create/Update Entity Files**:
   - For each identified entity, create or update its entity file
   - Populate with information from the chunks where it appears
   - Establish relationships with other entities

3. **Build Relationship Graph**:
   - Aggregate relationships from all entity files
   - Create a unified graph for visualization/navigation
