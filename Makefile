.PHONY: hooks

# Install the versioned hooks in .githooks/ for this clone (worktrees included).
# Run once after cloning.
hooks:
	git config core.hooksPath .githooks
