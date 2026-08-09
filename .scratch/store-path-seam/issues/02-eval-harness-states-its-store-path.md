# 02 — The evaluation harness states its store path instead of pinning the working directory

**Status:** ready-for-agent

**Blocked by:** 01

## What to build

The evaluation harness already knows where the store should live — it has a
resolver that pins the location against the repository root and honours the
`MYTH_INDEX_DIR` override, and that resolver is already tested. What it cannot
do today is *tell* `RAGSystem` about it. So both the index builder and the live
arm builder each wrap construction in a working-directory swap and hope the
relative literal inside resolves to the right place.

Now that ticket 01 lets a caller state the store location, both callers should
pass the resolved path directly and stop swapping the working directory.

This fixes a live defect, not just an ugliness. On the live evaluation path
`MYTH_INDEX_DIR` is currently accepted, logged as "index in use", and then
silently ignored — nothing passes it through to the system that opens the store.
A developer who sets it can therefore build one store and measure a different
one, with the log telling them the wrong thing.

Both call sites move in this ticket. They are the same one-line change, and
leaving one behind would mean the index builder and the live path disagree about
where the store is.

## Acceptance criteria

- [ ] Building an index no longer changes the process working directory.
- [ ] Building the live evaluation arms no longer changes the process working
      directory.
- [ ] Setting `MYTH_INDEX_DIR` and building an index puts the store in the
      specified directory.
- [ ] Setting `MYTH_INDEX_DIR` and running the live evaluation path reads from
      that same directory — the override is honoured end to end, and the
      "index in use" line reports the directory actually opened.
- [ ] With no override set, both paths behave exactly as they do today.
- [ ] The existing test suite passes unchanged, including the guards that keep
      the heavy retrieval stack out of the lightweight modules.
