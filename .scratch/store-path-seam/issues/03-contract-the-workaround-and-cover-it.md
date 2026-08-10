# 03 — Contract: retire the working-directory workaround and cover the seam

**Status:** ready-for-agent

**Blocked by:** 02

## What to build

The **contract** step. With every caller stating its store path (ticket 02), the
working-directory workaround has no remaining users and can be removed rather
than relocated. Then lock the seam shut with the test that was impossible to
write before.

Three parts.

**Retire the last workaround.** The unanswerable-question verification script
under `.scratch/rag-evaluation-harness/` carries the same working-directory swap
as the two production callers did. It already imports the path resolver — it just
never passes it. Move it over the same way, so no copy of the workaround survives
anywhere in the repository.

**Retire the compensating documentation.** The index module's docstring, and the
comments at each former workaround site, describe pinning the working directory
as the fix for a store that lands wherever the process started. That is no longer
true and will mislead the next reader. Rewrite them to describe what the code now
does. Leave the *separate* stale-index problem described in that docstring alone —
folding chunker parameters into the index key is a different defect and is still
real.

**Cover the seam.** Add the first test that constructs a `RAGSystem` against a
temporary store directory and asserts it reads and writes there. This is the
payoff for the whole sequence: today no test can touch `RAGSystem` at all,
because constructing one writes into the developer's real store. That is why the
production retrieval system has no tests, and why the existing suite's only
mention of it is an assertion that it stays unimported.

Keep that assertion working. The existing guards that keep the heavy stack out of
the lightweight modules are correct and must stay green — the new test is the
place where the heavy stack is allowed, and it should be marked so it can be
deselected on a machine without the models.

## Acceptance criteria

- [ ] No working-directory swap remains anywhere in the repository for the
      purpose of locating the store — production, harness, or scratch scripts.
- [ ] The unanswerable-question verification script runs correctly from a
      directory other than the repository root.
- [ ] The index module's docstring and the former workaround comments describe
      the current behaviour; the unrelated stale-index note is preserved.
- [ ] A test constructs a `RAGSystem` against a temporary directory and asserts
      both the vector store and the chunk cache land there.
- [ ] That test leaves the developer's real store untouched — running the suite
      does not create or modify it.
- [ ] The test is marked so it can be deselected where the models are
      unavailable, and the suite still passes with it deselected.
- [ ] The existing "no heavy imports" guards pass unchanged.
