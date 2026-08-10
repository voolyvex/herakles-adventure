# 03 — Contract: the runner stops carrying pooling's field

**Status:** ready-for-agent

**Blocked by:** 02

## What to build

The **contract** step. With every caller on the narrow form, the old one has no
users and can be withdrawn rather than left standing beside it.

Two things go.

**The wide parameter.** Pooling stops accepting whole evaluated arms. Once it
does, the direction of the dependency is settled: pooling reads a shape it
defines, rather than reaching into one the runner happens to expose.

**The helper.** The pool suite's manufactured-arms helper loses its last caller
in ticket 02 and is deleted here.

Then settle the field the whole sequence is about. The runner retains each
question's retrieved passages, populates them, and never reads them — pooling is
the only reader, and the runner's own serialisation deliberately leaves them
out. Decide, and record in the code, which of two things that field now is: part
of what an evaluation returns because retrieved passages are worth having, or a
handover to pooling that should be named as such. Either is defensible. What is
no longer acceptable is a field whose shape is set by another module's needs and
whose docstring has to explain the coupling.

The retained-items property must survive whichever way it goes: pooling reads
one evaluation pass, and must not pay for a second round of retrieval.

Finally, retire the compensating documentation. The docstrings that explain why
the runner carries a field for pooling's benefit, and why pooling imports the
depth from the runner, describe arrangements that no longer exist. Rewrite them
to describe what the code now does.

## Acceptance criteria

- [ ] Pooling no longer accepts whole evaluated arms; the narrow form is the only
      way in.
- [ ] The pool suite's manufactured-arms helper is gone.
- [ ] The runner's retained retrieved passages are either named as a handover to
      pooling or justified on their own terms, and the code says which.
- [ ] Pooling still reads a single evaluation pass — building a pool triggers no
      further retrieval.
- [ ] The runner's serialised results are unchanged: retrieved passages stay out
      of the results artefact.
- [ ] Pool output for the same inputs remains byte-identical to what the pre-
      sequence revision produced.
- [ ] Docstrings describing the old coupling are rewritten to describe current
      behaviour.
- [ ] The full test suite passes.
