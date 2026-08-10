.PHONY: hooks fmt lint

# ruff replaces black and isort both, so it is the only formatter here — see
# docs/agents/formatting.md. Prefer a ruff already on PATH (`uv tool install
# ruff`) and fall back to a pinned uvx, which needs the network on a cold cache.
# The pin is deliberate: an unpinned `uvx ruff` resolves to whatever is newest
# at run time, and a formatter whose version floats reformats the tree
# differently on two machines. Raise both this and the floor in pyproject.toml
# together.
RUFF ?= $(shell command -v ruff 2>/dev/null || echo "uvx ruff@0.16.2")

# Install the versioned hooks in .githooks/ for this clone (worktrees included).
# Run once after cloning.
hooks:
	git config core.hooksPath .githooks

# Format in place, then apply the lint fixes ruff considers safe.
fmt:
	$(RUFF) format .
	$(RUFF) check --fix .

# Report without changing anything. Same two passes as fmt, which is what makes
# a clean `make lint` mean `make fmt` would be a no-op.
#
# This exits non-zero today, and that is not a broken target: the code predates
# having a formatter, so a couple dozen files and several dozen findings are
# already waiting. Lint the files you touched rather than the whole tree until
# that backlog drains — running `make fmt` over everything would bury whatever
# you were actually asked to do in a repo-wide diff.
lint:
	$(RUFF) format --check .
	$(RUFF) check .
