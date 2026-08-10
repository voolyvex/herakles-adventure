# 02 — Tidy the pool module against house style

**Status:** ready-for-agent

**Blocked by:** None. Independent of 01, but touches the same file — do 01
first if both are picked up, since 01 is what gates #10.

Standards-axis findings from the same review as ticket 01. None of these change
behaviour or affect the gold set. They are here so the module reads like the rest
of the package rather than like something bolted on.

## 1. Bare `tuple` hides the shape it carries

```python
def _outcomes_by_question(arms: Sequence[ArmResult]) -> List[tuple]:
def _pool_one_question(question: Question, indexed_arms: Sequence[tuple], depth: int)
```

The real type is `Tuple[str, Dict[str, QuestionOutcome]]`. `Tuple` is not even
imported. A reader of `_pool_one_question` cannot learn from the signature that
`indexed_arms` yields `(arm_name, outcomes)` pairs — they have to read the body
to find out.

Every parameterised generic elsewhere in the package is spelled out:
`fakes.py` has `Optional[Mapping[str, Sequence[str]]]`, `runner.py` has
`metrics: Dict[str, float]`. (`nltk_resources.py`'s `REQUIRED_RESOURCES:
Sequence[tuple]` is a module constant, not a signature — not a precedent.)

Spell the type out. If the annotation reads badly at full length, that is a
signal the pair wants to be a small dataclass — an `IndexedArm` holding a name
and its outcomes-by-question map — which would also make the parameter name
`indexed_arms` honest. Either fix is acceptable; the bare `tuple` is not.

## 2. In-body imports that defer nothing

Three imports sit inside function or test bodies:

- `from myth_eval.retrieval import RetrievedItem` (test body)
- `from myth_eval.cli import main` (test body)
- `from myth_eval.pool import build_pool` (inside `cli.main`)

None of them defer anything heavy. `myth_eval.pool` imports only `json`,
`dataclasses`, `pathlib`, `dataset` and `runner`; `myth_eval.retrieval` is
already in the graph via `test_pool → pool → runner → retrieval`.

This matters because in this package an in-body import is a load-bearing signal,
not a formatting choice. `cli.build_live_arms` uses it to keep torch and chromadb
out of the default import graph, which is what lets the no-heavy-imports guards
mean anything. Copying the idiom without the reason tells a reader `pool` is
expensive when it is not. Move all three to module level. The only in-body import
in the existing tests is `import sys` inside the guard itself.

## 3. Two sentinels riding one CLI argument

```python
parser.add_argument("--pool-output", default=None, nargs="?", const="")
...
pool_path = Path(args.pool_output) if args.pool_output else default_pool_path()
```

`None` means "do not pool" and `""` means "pool to the default path" — two
distinct decisions carried by one string. It works, but the empty string as a
sentinel is invisible at the call site.

A separate boolean flag for "pool at all" plus a path argument for "where" would
say it plainly. Low priority; only worth doing if the CLI is being touched
anyway.

## 4. A test name that overclaims

`test_the_default_path_sits_beside_the_dataset` asserts only that the filename is
`candidate_pool.json` and its parent directory is named `eval_data`. It never
compares against `default_dataset_path().parent`, so it does not test the
"beside the dataset" relationship its name promises — if the dataset moved, this
test would still pass.

Either assert the actual relationship, or rename it to what it checks.

## Acceptance criteria

- [ ] No bare `tuple` annotations remain in `pool.py`; the `(arm_name, outcomes)`
      shape is legible from the signatures alone.
- [ ] The three non-deferring in-body imports are at module level, and the
      no-heavy-imports guards still pass — including `pytest -m heavy`, which is
      what would catch a mistake here.
- [ ] `test_the_default_path_sits_beside_the_dataset` either tests the
      relationship or is renamed.
- [ ] `pytest` passes with no test removed or weakened.

## Checked and cleared

Recorded so the next reader does not re-investigate:

- **The `heavy` marker rule holds.** `test_pool.py`'s guard adds `"torch"` to the
  forbidden tuple, a superset of `test_runner.py`'s three. Nothing in the default
  suite outside `test_rag_store_location.py` imports torch.
- **Duplicated guard loops across test modules are the documented design**, not
  Duplicated Code — `pyproject.toml`'s marker comment says "several tests assert
  the heavy modules are absent from `sys.modules`".
- **`CandidatePool.load -> Dict[str, Any]`** matches `EvaluationResults.load`
  exactly. Convention-following, not a Mysterious Name.
- **No dead code or leftover scaffolding** in `test_pool.py`. The
  `arm.outcomes[0].items = [...]` mutation is justified by its comment: the
  fake's own passages are too short to exercise excerpt trimming.

## Comments

Raised from `/code-review` against `0f63202...HEAD` on the `feat/candidate-pool`
branch, Standards axis. Split from ticket 01 because these are style-only and
must not gate the manual grading sitting.
