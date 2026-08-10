# Formatting and linting

This repo uses **ruff** and nothing else. `ruff format` is the formatter and `ruff check` is the linter, including the import sorting that would otherwise be isort's job.

Do not reach for black, isort, or flake8. They are not installed and are not listed as dependencies, so running them means installing a tool this repo has decided against — and black and ruff disagree in enough small ways that running both leaves the tree churning between two formats.

## Running it

ruff needs no install of its own if `uv` is present:

| Command | What it does |
| ------- | ------------ |
| `make lint` | Report formatting drift and lint findings. Changes nothing. |
| `make fmt` | Format in place, then apply the safe lint fixes. |
| `ruff check path/to/file.py` | Lint a single file. |
| `ruff format path/to/file.py` | Format a single file. |

`uv tool install ruff` puts `ruff` on `PATH` permanently, which is worth doing once — the project venv is per-worktree and gitignored, so a tool installed into it does not follow you to the next branch. The Makefile uses that binary if it finds one and falls back to a pinned `uvx ruff@<version>` otherwise; the pin is what stops two machines formatting the tree two ways.

Note that `ruff` is declared in the `dev` extra but the project venv does not necessarily have it — `uv pip install -e '.[dev]'` installs it there if you would rather not have it on `PATH`.

## The existing backlog

`make lint` exits non-zero today: a couple dozen files predate having a formatter, and there are several dozen lint findings — unused imports and unsorted import blocks, concentrated in `rag_system.py`, `god_chat.py` and `agents/`.

This is known, and it is not an invitation to run `make fmt` over the whole tree. A repo-wide reformat buries whatever you were actually asked to do in a few hundred lines of noise. **Lint the files you touched**, leave the rest, and let the backlog drain as those files get edited for other reasons.

At least one of those findings is load-bearing, so read before you fix rather than trusting `--fix`: `tests/test_runner.py` imports `evaluate_arm` without calling it, and that import *is* the assertion — it checks the symbol exists and is importable without dragging in torch. Deleting it silently weakens the test.

The rest of the unused imports do look genuinely dead: nothing outside `rag_system.py` and `god_chat.py` imports the `utils.name_mapping` names they pull in, so those are safe to drop when you are already editing the file.

## Settings

All of it lives in `[tool.ruff]` in `pyproject.toml`, with the reasoning next to each choice. The short version: line length 88 because the maintained code is already hand-wrapped to it; `E501` off because the formatter owns line length and reporting it twice is noise; `E402` exempted in three modules whose import order is load-bearing; `.scratch/` excluded because tickets are prose, not source.

The selected rules are `E`, `F` and `I` — a floor, not a ceiling. Widen it once the backlog above is paid down, not before.
