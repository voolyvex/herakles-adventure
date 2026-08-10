# 02 — Move pooling's callers to the narrow form

**Status:** ready-for-agent

**Blocked by:** 01

## What to build

The **migrate** step. With both forms working, move everyone who builds a pool
onto the narrow one.

There are two kinds of caller. The command builds a pool from the run that has
just happened, and must keep doing so without paying for a second round of
retrieval — that property is why the runner retains retrieved items at all, and
it must survive this ticket intact. The pool suite is the other caller, and it
is the one that gets shorter: its tests can construct retrievals directly
instead of running an evaluation to obtain data they never read.

The helper the pool suite uses to manufacture evaluated arms should have no
callers left when this ticket is done. Removing it is ticket 03's job, but if it
still has callers here, the migration is not finished.

Keep the tests that genuinely exercise pooling behaviour — the blindness of the
grading view to which arm found a candidate, the deduplication by document, the
partition of unanswerable questions, the reproducibility of the ordering. Those
are the reason the suite is long, and they are earning it. What goes is the
setup ceremony in front of them.

## Acceptance criteria

- [ ] The command builds its pool through the narrow form, still reusing the run
      that has already happened and costing no extra retrieval.
- [ ] Every pooling test constructs its inputs directly, with no evaluation run.
- [ ] The helper that manufactures evaluated arms for the pool suite has no
      remaining callers.
- [ ] Pool output is unchanged: for the same inputs, the generated artefact is
      byte-identical to what the previous revision produced.
- [ ] The depth-clamping behaviour is still exercised by a test that drives the
      command end to end.
- [ ] The full test suite passes.
