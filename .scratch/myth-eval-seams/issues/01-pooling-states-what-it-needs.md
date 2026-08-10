# 01 — Pooling states what it needs from a run

**Status:** ready-for-agent

**Blocked by:** None — can start immediately

## What to build

Building a candidate pool needs two things from an evaluation: which arm found a
document, and what that arm retrieved for each question. It does not need the
scores, the latency percentiles, the per-stratum breakdowns or the diagnostics —
and today it asks for all of them, because it takes a whole evaluated arm.

The cost lands on whoever writes a pooling test. To pool anything at all you
must first run an evaluation, which means scoring every question and aggregating
metrics nobody is about to read. The pool suite carries a helper that exists for
no other reason.

Make pooling say what it actually needs: an arm's name, and that arm's retrieved
items per question. Keep accepting evaluated arms exactly as today, so nothing
downstream has to move yet.

This is the **expand** step of an expand–contract sequence. The narrow form is
added beside the existing one and both work; callers move over in ticket 02, and
the old form is withdrawn in ticket 03.

**Pooling depth moves in this ticket too.** The depth is a pooling concept, but
it is defined by the runner and imported backwards, and the rule that protects
it — that a deeper retrieval run must never widen the pool past the spec's depth
— lives in the command, furthest from the concept it guards. Answering "how deep
do we pool, and what happens if the requested depth differs" currently means
reading three modules. Bring the depth and the clamping rule to the module that
owns them, and let the runner and the command import from there.

Do both together, or neither. The narrow form is what makes pooling stop
depending on the runner; leaving the depth behind would keep the import pointing
the wrong way and leave the sequence half-done.

## Acceptance criteria

- [ ] A caller can build a pool from arm names and their retrieved items, with
      no evaluation run and no metrics computed.
- [ ] Building a pool from evaluated arms still works and produces byte-identical
      output to today, for the same inputs.
- [ ] The pooling depth and the rule that clamps a requested depth to it are
      defined in the pooling module; the runner and the command read them from
      there.
- [ ] The pooling module no longer imports the depth from the runner.
- [ ] The warning that fires when a pool is built below the spec's depth still
      fires, with the same meaning.
- [ ] The full existing test suite passes unchanged, including the guards that
      keep the heavy stack out of the lightweight modules.
