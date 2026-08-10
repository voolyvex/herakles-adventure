# 01 — Close the rank leak in the grading sheet

**Status:** ready-for-agent

**Blocked by:** None. Blocks #10 (manual relevance grading) — see "Why this is
urgent" below.

Found by a two-axis review of the #9 candidate pool generator (PR #17,
commits `dc3eda4`, `f59dfcd`). Four defects, all in `myth_eval/pool.py` unless
noted. The first is the reason this ticket exists; the rest are cheap to fix
while the file is open.

## Why this is urgent

The pool generator is correct on all five of #9's acceptance criteria. This
ticket is not about those. It is about a property the module claims in its own
docstring and does not have — and the window to fix it closes the moment a
human starts grading.

Once a labeller has worked through a rank-ordered sheet, the bias is baked into
the gold set and cannot be removed after the fact. Regrading is the only remedy,
and regrading is the expensive manual sitting the whole harness exists to spend
only once. **Fix before #10 starts.**

## 1. The grading sheet leaks retriever rank through its ordering

`PooledCandidate.best_rank` is documented as "deliberately excluded from the
graded view", and `to_grading_dict()` does omit the key. But `_pool_one_question`
sorts on it:

```python
candidates = sorted(
    merged.values(),
    key=lambda candidate: (candidate.best_rank, candidate.source_document),
)
```

so ordinal position on the sheet *is* the rank. Omitting the field moved the
information from a key to a sequence; it did not remove it. Confirmed by running
the generator: an arm returning `b.md`(r0), `a.md`(r1), `c.md`(r2) emits graded
candidates in the order `['b.md', 'a.md', 'c.md']` — identical to the order in
`diagnostics.attribution`, which carries the ranks explicitly.

This contradicts two rules in `grading_rubric.md`:

> **Retriever score or rank.** Grading is blind to which arm(s) surfaced a
> candidate or how they ranked it. Grading by score would bias the gold set
> toward whichever arm generated the labels — the exact bias pooled judging
> exists to avoid.

> Grade candidates **independently**. Don't compare a candidate against other
> candidates in the pool before assigning it a grade.

A sheet ordered best-first invites exactly that comparison, and a labeller
working down it will read the top entries as the system's favourites.

The original rationale for rank-ordering — "so the strongest candidates are read
first" — is itself the defect, not a benefit to preserve. Sort the **graded
view** on `source_document` alone. That is already a total order, because
`source_document` is the dedup key and is therefore unique within a question, so
reproducibility (#9's AC5) is unaffected. Keep `best_rank` in
`diagnostics.attribution`, where it is genuinely useful and where no grading
decision reads it.

Note the ordering guarantee this ticket must not break: candidate order must
still not depend on arm order. `f59dfcd` made the excerpt choice total via
`(item.rank, item.chunk_id)` for exactly this reason. That tie-break governs
which *chunk* represents a document and stays as it is; only the sort of the
emitted candidate list changes.

## 2. A shallow `--k` silently emits a non-conforming pool

`cli.py` clamps the pooling depth:

```python
pool = build_pool(results.arms, dataset, depth=min(args.k, POOL_DEPTH))
```

This correctly stops `--k 20` widening the pool past ten. It does nothing for
`--k 3`, which writes a top-3 pool to the default path with no warning.

`spec.md` fixes the depth in two places — "take the union of their **top-10**
results" and "10 is the pooling depth" — and adds the consequence that makes an
under-depth pool permanent:

> Anything outside the pool scores zero.

So a pool built at depth 3 zeroes documents that a conforming pool would have
contained, and the resulting labels are wrong in a way no later run can detect.
The `pool_depth` provenance field records this, but a field a reader may never
check is thin protection against a silently corrupt gold set.

Log a warning when the effective depth is below `POOL_DEPTH`, naming the depth
and saying the pool will not conform to the spec. A warning rather than a
refusal: a shallow pool is legitimate for a smoke test, it just must not be
mistaken for the real one.

## 3. `default_pool_path()`'s docstring is false

```python
def default_pool_path() -> Path:
    """The committed candidate pool file."""
```

`git ls-files eval_data` lists only `questions.json`. The pool is deliberately
not committed — the only pool that can be generated without a live index comes
from scripted fake arms, and one sitting at the real path would read as genuine.
The absence is right; the docstring is wrong. Say what it is: the default
location the generated pool is written to, and that generating it needs a built
index.

## 4. `EXCERPT_CHARS` is public but missing from `__all__`

`tests/test_pool.py` imports it, which makes it public whatever the intent. The
package convention is to declare public constants — `runner.py` lists
`K_VALUES`, `GATE_K` and `POOL_DEPTH`; `dataset.py` lists `GRADE_LABELS`. Add it.

## Acceptance criteria

- [ ] The graded candidate list is ordered by `source_document` alone; its
      ordering carries no rank information.
- [ ] A test asserts the emitted grading order differs from the rank order when
      the two disagree — not merely that `best_rank` is absent as a key. The
      current test checks only key absence, which is why this defect survived.
- [ ] `best_rank` is still reported under `diagnostics.attribution`.
- [ ] Candidate ordering still does not depend on arm order, and two runs over
      the same inputs still produce byte-identical output.
- [ ] Running the generator with an effective pool depth below 10 logs a warning
      naming the depth; a test covers it.
- [ ] `--k 20` still pools at depth 10, and `--k 3` still pools at depth 3.
- [ ] `default_pool_path()`'s docstring describes a generated file, not a
      committed one.
- [ ] `EXCERPT_CHARS` appears in `pool.py`'s `__all__`.
- [ ] `pytest` passes with no test removed or weakened; `pytest -m heavy` passes.

## Out of scope

- **Committing `eval_data/candidate_pool.json`.** Generating the real pool needs
  the live index and is the first step of #10.
- **Anything consuming the pool.** Feeding graded labels back into the dataset is
  #10. `metrics.py` already scores unjudged documents zero via
  `grades.get(document, 0)`, so the "outside the pool scores zero" half of the
  spec is live and needs nothing here.
- **`GRADING_INSTRUCTIONS` duplicating #10's decision tree.** The review flagged
  it as scope creep that can drift from `grading_rubric.md`. Left as-is
  deliberately: a grading sheet that does not state its own scale is worse than
  one carrying a summary, and the text points at the rubric as the authority.
  Revisit only if the two actually diverge.

## Comments

Raised from `/code-review` against `0f63202...HEAD` on the `feat/candidate-pool`
branch. Findings 1 and 3 correct claims made in PR #17's own description, which
asserted the grading view was blind to rank. The omitted key was real; the
ordering was not checked.
