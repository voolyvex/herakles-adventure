# 05 — The fused candidate pool becomes public

**Status:** ready-for-agent

**Blocked by:** None — can start immediately

## What to build

The reranked arm needs the hybrid arm's fused candidate pool — unsorted, and
deeper than what the hybrid arm returns to a caller. The hybrid arm's interface
offers only a retrieval that returns the top results already sorted and
renumbered, so the reranked arm reaches past it into a private attribute and
then re-sorts the result itself.

The consequence is that the fusion rule is implemented twice. Both sites sort
the merged candidates by raw score — and that sort carries a defect the harness
is deliberately preserving: cosine similarity and BM25 are not on a common
scale, so BM25 dominates the ordering for reasons that have nothing to do with
relevance. The first baseline is meant to capture that defect so the follow-up
fix can be demonstrated as a measured improvement rather than asserted.

That plan assumes the fix is a single edit. Today it is two, and the second one
is easy to miss, because it lives in a different class and was written to look
like a local detail.

Give the hybrid arm the method its interface is missing: the fused,
score-ordered candidate pool. The reranked arm asks for it through the
interface. The fusion rule exists once.

**The defect stays.** This ticket changes where the rule lives, not what it
does. Fusion behaviour must be identical before and after — same ordering, same
candidates, same defect. Anything else would forfeit the baseline the harness
exists to capture.

## Acceptance criteria

- [ ] The hybrid arm exposes its fused candidate pool through its interface.
- [ ] The reranked arm obtains its pool through that interface and does not
      touch any private attribute of the hybrid arm.
- [ ] The sort that fuses dense and sparse candidates appears once.
- [ ] Fusion behaviour is unchanged: for the same inputs, both arms return the
      same candidates in the same order as before this change.
- [ ] The score-scale defect is still present, and the comment explaining why it
      is preserved still sits with the code that implements it.
- [ ] A test exercises fusion through the new interface without involving a
      reranker.
- [ ] The full test suite passes.
