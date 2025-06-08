# Myth-RPG

A mythology-themed text-based RPG where you can converse with Greek gods.

## Quick Start

1. **Install UV** (if not already installed):
   ```sh
   pip install uv
   ```

2. **Set up the environment**:
   ```sh
   # Create and activate virtual environment
   uv venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac

   # Install dependencies
   uv pip install -e .
   ```

3. **Run the game**:
   ```sh
   myth-rpg
   # or
   python -m god_chat
   ```

## Development

- Install development dependencies:
  ```sh
  uv pip install -e ".[dev]"
  ```

- Run tests:
  ```sh
  pytest
  ```

## License

MIT

----
Created by voolyvex
