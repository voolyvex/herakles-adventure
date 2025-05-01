import json
import jsonschema
from pathlib import Path

def validate_poets():
    schema = json.loads(Path("lore_entities/poet_core.json").read_text())
    poets_dir = Path("entities/poets")
    
    print(f"Validating against schema version {schema.get('$schema', 'unknown')}")
    
    for poet_file in poets_dir.glob("*.json"):
        try:
            data = json.loads(poet_file.read_text())
            jsonschema.validate(data, schema)
            
            # Check optional field completeness
            optional_fields = set(schema['properties'].keys()) - set(schema.get('required', []))
            missing_optional = [f for f in optional_fields if f not in data]
            
            if missing_optional:
                print(f"✓ Valid but missing optional: {poet_file.name} (missing: {', '.join(missing_optional)})")
            else:
                print(f"✓ Fully valid: {poet_file.name}")
                
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON: {poet_file.name}\n  {str(e)}")
        except jsonschema.ValidationError as e:
            print(f"✗ Schema violation: {poet_file.name}\n  {e.message}")

if __name__ == "__main__":
    validate_poets()
