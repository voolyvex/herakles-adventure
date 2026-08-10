# 04 — One assembly for both the real and the scripted matrix

**Status:** ready-for-agent

**Blocked by:** None — can start immediately

## What to build

The configuration matrix is assembled twice. The real path builds it from the
live retrieval stack through the arms module; the scripted path hand-assembles
the same four arms inline. They agree today — same names, same types, same
fan-out, same pool size — but only because the same literals are written out in
both places.

The command's own docstring claims that running against scripted retrievers
"runs the identical code path". That is true of the runner, the metrics and the
pooling, and false of assembly, which is the one part that differs.

Make it true. One module assembles the matrix; the command's only job is to
choose which adapters to hand it — the live stack, or the scripted ones. That is
two adapters at one seam, which is what makes the seam worth having.

There is one real asymmetry to preserve, not paper over. The real assembly drops
the sparse arm when it is unavailable; the scripted path has no counterpart
because it always has one. After this ticket, one rule governs both.

**The guard is the point of this ticket.** When the sparse retriever fails to
initialise, the command refuses to produce results rather than silently scoring
the sparse and hybrid arms zero for reasons unrelated to retrieval quality. It
is the loudest safety behaviour in the harness, it exists precisely because
silent degradation would poison the committed baseline, and no test has ever
executed it — because the only place it lives cannot be constructed without the
heavy stack. Once assembly is one testable module, that guard becomes reachable.
Reach it.

Also fold in the canonical arm-name list, which is currently declared and then
consumed by nothing while the same four strings are produced by constructor
defaults and hardcoded again in the runner suite. One declaration, actually
used.

## Acceptance criteria

- [ ] The real and scripted matrices are assembled by the same code; the command
      only chooses which adapters to supply.
- [ ] Running against scripted retrievers produces the same arm names, types and
      ordering as before this change.
- [ ] The rule that drops the sparse arm when it is unavailable applies to both
      paths.
- [ ] A test executes the refusal that fires when the sparse retriever fails to
      initialise, and asserts it refuses rather than reporting zeros.
- [ ] That test runs without the heavy stack — no torch, chromadb or embedding
      model — and the existing "no heavy imports" guards still pass.
- [ ] The canonical arm-name list has at least one real consumer, and the arm
      names are not written out separately in the runner suite.
- [ ] The command's docstring claim about the scripted path is accurate.
- [ ] The full test suite passes.
