# 01 — RAGSystem accepts where its store lives

**Status:** ready-for-agent

**Blocked by:** None — can start immediately

## What to build

Today, where the vector store lands is not something a caller can state — it is a
consequence of which directory the process happened to start in. `RAGSystem`
opens its store with a bare relative literal, and resolves its lore chunk cache
the same way. Two callers running from different directories get two unrelated
stores.

Make the store location part of what a caller says when they construct a
`RAGSystem`, defaulting to exactly today's behaviour when they say nothing.

This is the **expand** step of an expand–contract sequence: the new form is added
beside the old behaviour, so nothing downstream has to change yet and every
existing test stays green. The callers move over in ticket 02; the workarounds
are deleted in ticket 03.

**The store directory and the lore cache must move together.** There are two
relative literals, not one: the Chroma client path and the chunk cache path. All
three cache readers and writers reach the cache through a single accessor, so
this is a small change — but if only the client path becomes configurable, a
build would write the store to one directory and its chunk cache to another,
which is worse than the current situation. Both, or neither.

## Acceptance criteria

- [ ] A caller can construct a `RAGSystem` that puts its store in a directory of
      their choosing, including an absolute path outside the repository.
- [ ] A caller who says nothing gets exactly today's behaviour — the same
      directory, resolved the same way.
- [ ] The chunk cache lands inside the same store directory as the vector store,
      whichever directory that is.
- [ ] No relative store literals remain in the RAG system module.
- [ ] The full existing test suite passes unchanged. In particular the pinned
      -location tests still hold: the default store resolves under the repository
      root, and that resolution does not depend on the working directory.
- [ ] The game still runs and answers a question, started from the repository
      root as before.
