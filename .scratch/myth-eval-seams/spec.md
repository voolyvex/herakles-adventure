# myth_eval seams

Three deepening candidates from an architecture review of `myth_eval/`, the
hot spot across the harness's build-out. Each is about **where a seam sits**,
not what is behind it — every module involved already passes the deletion test.

Out of scope, and deliberately: the store-path seam (closed by its own ticket
set) and the retrieval protocol, which has two adapters and is already deep.

## The three

**Pooling asks for more than it needs.** Building a candidate pool requires an
arm's name and what it retrieved per question. It takes a whole evaluated arm —
metrics, latency percentiles, per-stratum breakdowns, diagnostics — and reads
two fields of six. The cost lands on the pool suite, where every test runs an
evaluation to obtain data it never reads. The coupling runs both ways: pooling
imports its own depth from the runner, and the runner carries a field of
retrieved passages that only pooling reads. Tickets 01–03, as expand–migrate–
contract.

**The matrix is assembled twice.** The real path and the scripted path build the
same four arms by different routes, agreeing only because the literals are
duplicated in both. The command's docstring claims the scripted path is the
identical code path; it is, except for assembly. The casualty is the harness's
loudest safety guard — the refusal to report results when the sparse retriever
fails to initialise — which no test has ever executed, because the only place it
lives cannot be constructed without the heavy stack. Ticket 04.

**Fusion is implemented twice.** The reranked arm reaches into the hybrid arm's
private candidate assembly and re-sorts it, duplicating the fusion sort. That
sort carries the score-scale defect the harness preserves on purpose, so the
planned fix — meant to be demonstrated as a measured improvement — is a two-site
edit that reads like a one-site edit. Ticket 05.

## Sequence

01 → 02 → 03 is a chain. 04 and 05 are independent of it and of each other; all
three of 01, 04 and 05 can start immediately.

01 and 04 both touch the command module, in different functions. That is a merge
conflict to resolve, not a blocking edge.
