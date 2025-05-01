import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any


def snake_to_title(text: str) -> str:
    """Convert a filename-like string (e.g., '001_INTRODUCTION') to a title string
    (e.g., 'Introduction').
    """
    # Remove leading digits and underscores/hyphens (chapter numbers)
    stripped = re.sub(r"^[0-9]+[_-]*", "", text)
    # Replace underscores/hyphens with spaces and title-case the result
    return stripped.replace("_", " ").replace("-", " ").title()


def build_base_entity(md_filename: str) -> Dict[str, Any]:
    """Return the default JSON structure for a lore entity derived from an .md chunk."""
    base_id = Path(md_filename).stem  # e.g. '001_INTRODUCTION'
    return {
        "id": base_id,
        "title": snake_to_title(base_id),
        "type": "lore_chunk",  # can be refined later (character, place, etc.)
        "related_chunks": [md_filename],
        "entities": [],  # referenced named entities inside the chunk
        "categories": [],  # taxonomy tags (e.g., myth, character, location)
        "relationships": [],  # explicit relationships built in future passes
        "summary": "",  # short abstract to be filled later
        "notes": "",  # any editorial notes
    }


def init_base_entities(chunks_dir: Path, entities_dir: Path, overwrite: bool = False) -> None:
    """Create a JSON file in `entities_dir` for every .md file in `chunks_dir`."""
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    entities_dir.mkdir(parents=True, exist_ok=True)

    md_files: List[Path] = sorted(chunks_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {chunks_dir}")
        return

    for md_path in md_files:
        entity_data = build_base_entity(md_path.name)
        json_path = entities_dir / f"{md_path.stem}.json"

        if json_path.exists() and not overwrite:
            print(f"Skipping existing file: {json_path}")
            continue

        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(entity_data, fp, ensure_ascii=False, indent=2)
        print(f"Created {json_path.relative_to(entities_dir.parent)}")


def classify_entity_type(title: str, id_: str) -> str:
    """Heuristic type assignment for lore entities."""
    title_lower = title.lower()
    id_lower = id_.lower()
    # Short keyword lists for brevity
    character_kw = ['zeus', 'hercules', 'athena', 'apollo', 'hera', 'poseidon', 'perseus', 'orpheus', 'odysseus', 'ulysses', 'achilles', 'hector', 'medusa', 'minerva', 'venus', 'jupiter', 'diana', 'bacchus', 'midas', 'pandora', 'prometheus', 'daedalus', 'icarus', 'proserpine', 'scylla', 'charybdis', 'theseus', 'persephone', 'hephaestus', 'ares', 'artemis', 'demeter', 'hermes', 'pan', 'janus', 'saturn', 'pluto', 'mars', 'cupid', 'psyche', 'adonis', 'hyacinthus']
    group_kw = ['divinities', 'pantheon', 'gods', 'heroes', 'giants', 'centaurs', 'nymphs', 'monsters', 'muses', 'deities']
    place_kw = ['olympus', 'delphi', 'elysium', 'troy', 'rome', 'athens', 'thebes', 'underworld', 'hades', 'island', 'mount', 'lake', 'river', 'city', 'temple', 'palace', 'forest', 'cave', 'garden', 'sea', 'ocean', 'country', 'region', 'empire', 'kingdom', 'land', 'realm']
    item_kw = ['sword', 'shield', 'helmet', 'armor', 'bow', 'arrow', 'ring', 'staff', 'wand', 'fleece', 'apple', 'box', 'horn', 'cup', 'chariot', 'statue', 'gift', 'treasure', 'relic', 'artifact', 'amulet', 'jewel', 'gem', 'stone', 'tablet', 'book', 'scroll', 'mirror', 'torch', 'crown', 'throne', 'lyre', 'harp', 'flute', 'trident', 'club', 'net', 'rope', 'chain', 'belt', 'boots', 'sandals', 'mask', 'coin', 'key', 'seal', 'scepter', 'orb', 'rod', 'spear', 'dagger', 'blade', 'scabbard', 'scythe', 'lance', 'chalice', 'potion', 'elixir']
    concept_kw = ['fate', 'oracle', 'destiny', 'prophecy', 'myth', 'legend', 'virtue', 'vice', 'wisdom', 'war', 'love', 'death', 'life', 'immortality', 'sacrifice', 'revenge', 'justice', 'honor', 'glory', 'peace', 'chaos', 'order', 'creation', 'destruction', 'rebirth']
    # Check keywords
    for kw in character_kw:
        if kw in title_lower or kw in id_lower:
            return 'character'
    for kw in group_kw:
        if kw in title_lower or kw in id_lower:
            return 'group'
    for kw in place_kw:
        if kw in title_lower or kw in id_lower:
            return 'place'
    for kw in item_kw:
        if kw in title_lower or kw in id_lower:
            return 'item'
    for kw in concept_kw:
        if kw in title_lower or kw in id_lower:
            return 'concept'
    return 'lore_chunk'


def classify_entity_types(entities_dir: Path, force: bool = False):
    """Update the 'type' field for all entities in entities_dir using heuristics.
    If force=True, always overwrite the type field.
    """
    json_files = list(entities_dir.glob("*.json"))
    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        title = data.get('title', '')
        id_ = data.get('id', '')
        new_type = classify_entity_type(title, id_)
        if force or data.get('type') != new_type:
            data['type'] = new_type
            with json_path.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
            print(f"Updated type for {json_path.name} to {new_type}")
        else:
            print(f"Type for {json_path.name} already {new_type}")


def reset_lore_chunk_types(entities_dir: Path):
    """Reset all entity files in the entities directory to have type='lore_chunk'.
    They are containers for lore text, not entity types themselves.
    """
    json_files = list(entities_dir.glob("*.json"))
    for json_path in json_files:
        # Skip template or special files
        if json_path.name.upper() in ["ENTITY_TEMPLATE.JSON", "TYPE_SYSTEM.MD", "PROPER_DATA_MODEL.MD"]:
            continue
            
        try:
            with json_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            # Reset to lore_chunk
            if data.get('type') != "lore_chunk":
                data['type'] = "lore_chunk"
                with json_path.open("w", encoding="utf-8") as fp:
                    json.dump(data, fp, ensure_ascii=False, indent=2)
                print(f"Reset {json_path.name} to type='lore_chunk'")
            else:
                print(f"{json_path.name} already type='lore_chunk'")
        except Exception as e:
            print(f"Error processing {json_path.name}: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Lore entity utilities")
    parser.add_argument(
        "--init-base-entities",
        action="store_true",
        help="Generate base entity JSON files corresponding to lore chunks.",
    )
    parser.add_argument(
        "--classify-types",
        action="store_true",
        help="Classify entity types for all JSONs in the entities directory.",
    )
    parser.add_argument(
        "--reset-types",
        action="store_true",
        help="Reset all lore JSON files to type='lore_chunk'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of type field even if already set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files if they already exist.",
    )
    parser.add_argument(
        "--chunks-dir",
        default="lore_chunks",
        help="Relative path to the directory containing markdown lore chunks.",
    )
    parser.add_argument(
        "--entities-dir",
        default="lore_entities",
        help="Relative path to the directory where entity JSONs will be stored.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    chunks_dir = (project_root / args.chunks_dir).resolve()
    entities_dir = (project_root / args.entities_dir).resolve()

    if args.init_base_entities:
        init_base_entities(chunks_dir, entities_dir, overwrite=args.overwrite)
    elif hasattr(args, 'classify_types') and args.classify_types:
        classify_entity_types(entities_dir, force=getattr(args, 'force', False))
    elif hasattr(args, 'reset_types') and args.reset_types:
        reset_lore_chunk_types(entities_dir)
    else:
        print("No action specified. Use --help for available commands.")


if __name__ == "__main__":
    main()