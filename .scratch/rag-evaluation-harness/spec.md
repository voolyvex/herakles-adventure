# Spec: RAG Evaluation Harness (MVP)

Status: ready-for-agent

## Problem Statement

The Herakles RAG system works, but nobody can say whether any given change to it
makes retrieval better or worse. There is dense retrieval, BM25 sparse retrieval,
a hybrid path, and a reranker — and no way to compare them beyond running a query
and forming an impression.

This has three concrete consequences:

- **Changes are unfalsifiable.** The outstanding embedding-model upgrade
  (`bge-small` → `nomic-embed-text-v1.5`) is documented as "Retrieval quality:
  Good → Better", but that claim is sourced from vendor material, never measured
  on this corpus. There is no way to accept or reject it.
- **Defects hide in plain sight.** The hybrid fusion step merges dense and sparse
  results and sorts them by a raw `score` field that mixes cosine similarity
  (bounded 0–1) with BM25 scores (unbounded). Nothing catches this, because
  nothing measures ranking quality.
- **Retrieval regressions are invisible to CI.** There is no CI at all. A change
  that quietly halves recall would merge exactly as easily as a typo fix.

The goal is to move from "the RAG demo seems good" to "I can show what improved,
what regressed, and what threshold I would require before promoting it."

## Solution

An evaluation harness that scores five retrieval configurations against one fixed,
graded, stratified question set and reports ranking quality and latency for each.

A single command runs the whole matrix. Results are written machine-readably and
compared against a committed baseline. CI fails a pull request when aggregate
ranking quality regresses beyond an empirically derived tolerance — so retrieval
quality becomes a merge gate, not a vibe.

The MVP is **retrieval-only**. It measures whether the right passages are found
and how they are ranked. It does not measure generated prose.

## User Stories

1. As a developer changing retrieval code, I want a single command that evaluates
   every retrieval configuration, so that I can see the effect of my change without
   assembling a test rig each time.
2. As a developer, I want the evaluation to run against a fixed, version-controlled
   question set, so that two runs a month apart are comparable.
3. As a developer, I want ranking quality reported as nDCG, so that a configuration
   that returns the right passage at rank 1 scores better than one that returns it
   at rank 9.
4. As a developer, I want Recall@K reported alongside nDCG, so that I have a number
   that is intuitively legible to someone who has not used nDCG before.
5. As a developer, I want MRR reported, so that I can see how quickly the first
   relevant result appears.
6. As a developer, I want per-configuration latency at p50 and p95, so that I can
   weigh a quality gain against the time it costs.
7. As a developer, I want to compare dense-only retrieval against BM25-only, so
   that I know which of the two arms is actually carrying the system.
8. As a developer, I want to evaluate the hybrid configuration, so that I can tell
   whether combining the arms beats either one alone.
9. As a developer, I want to evaluate hybrid-plus-reranker, so that I can quantify
   what the reranker is worth.
10. As a developer, I want a fifth configuration that applies the `god_context`
    filter, so that I measure the path production actually calls rather than only
    the idealised ones.
11. As a developer, I want the `god_context` arm to expose how often the filter is
    inert, so that a suspected data-quality problem becomes a measured fact.
12. As a developer, I want questions grouped into strata, so that I can see whether
    a change helps synthesis questions while leaving factual lookups alone.
13. As a developer, I want single-hop factual questions in the set, so that basic
    lookup quality is covered.
14. As a developer, I want multi-hop synthesis questions, so that configurations
    are tested on queries needing more than one passage.
15. As a developer, I want entity-centric questions, so that retrieval of specific
    named figures is measured.
16. As a developer, I want unanswerable questions drawn from mythologies outside
    the corpus, so that I can measure whether the system correctly returns nothing.
17. As a developer, I want every unanswerable question verified absent from the
    corpus before it is added, so that I am not penalising the system for finding
    material that genuinely exists.
18. As a developer, I want relevance graded on a three-point scale rather than
    binary, so that "directly answers" and "related background" are distinguished.
19. As a developer, I want ground truth labelled against source documents rather
    than chunk identifiers, so that re-chunking the corpus does not invalidate my
    labels.
20. As a developer, I want the chunker parameters recorded in the dataset as
    provenance, so that I know which chunking regime the labels were gathered under.
21. As a developer, I want the graded candidate pool drawn from the union of all
    configurations' results, so that the baseline does not silently favour whichever
    retriever happened to generate the labels.
22. As a developer, I want unjudged documents to score zero, so that pooled judging
    has well-defined behaviour on unseen results.
23. As a developer, I want a candidate pool generated for me automatically, so that
    my manual labelling session is one focused sitting against a prepared list.
24. As a developer, I want results written to a machine-readable file, so that they
    can be diffed, archived, and compared by tooling.
25. As a developer, I want an accepted baseline committed to the repository, so that
    every later run has something to compare against.
26. As a developer, I want the runner to report the delta against the baseline, so
    that I see the direction and size of a change immediately.
27. As a developer, I want the regression tolerance derived from repeated runs
    rather than guessed, so that the gate does not fire on ordinary noise.
28. As a developer, I want CI to fail on a genuine regression, so that quality
    problems are caught before merge rather than after.
29. As a developer, I want the gate to run only when retrieval-relevant paths
    change, so that unrelated pull requests are not slowed by it.
30. As a developer, I want gating on the aggregate figure only, so that a noisy
    stratum of eight questions cannot block a merge on its own.
31. As a developer, I want per-stratum numbers reported as diagnostic and marked
    indicative, so that I can read them for signal without mistaking them for gates.
32. As a developer accepting a deliberate regression, I want to update the baseline
    in an explicit commit, so that the decision is recorded rather than bypassed.
33. As a developer, I want the CI index build cached, so that the gate returns a
    verdict in reasonable time.
34. As a developer, I want the cache key to include chunker parameters, so that
    changing chunk size cannot silently reuse a stale index.
35. As a developer, I want a periodic cold rebuild, so that cache drift is caught
    even when no pull request forces it.
36. As a developer, I want an explicit index-building step with a pinned working
    directory, so that the vector store lands in a predictable location rather than
    wherever the process happened to start.
37. As a developer, I want dense and sparse retrievers behind one interface, so
    that the harness can call them uniformly.
38. As a developer, I want retrieved results normalised into one shape, so that
    metric code does not branch on which retriever produced a result.
39. As a developer, I want the interface adapter to wrap the existing agents without
    modifying their internals, so that the harness does not destabilise the working
    application.
40. As a developer, I want metric functions to be pure, so that they can be tested
    against hand-computed values without loading a model.
41. As a developer, I want to run the whole harness against a fake retriever, so
    that runner logic is testable without ChromaDB, embeddings, or a reranker.
42. As a future developer adding an Azure retriever, I want a stable retrieval
    interface, so that a new backend can be evaluated on the same dataset without
    touching the harness.
43. As a developer, I want the baseline to be established before the fusion defect
    is fixed, so that the fix can be demonstrated as a measured improvement.
44. As a developer, I want the deliberate decision to baseline a known defect
    recorded in the README, so that it reads as intent rather than oversight.
45. As a developer, I want the existing no-assertion smoke script renamed out of
    pytest's collection range, so that enabling CI does not trigger a multi-gigabyte
    model download.
46. As a developer, I want packaging fixed so the project installs on Linux, so
    that CI can install the package at all.
47. As a reviewer, I want a README recording the metrics, the rationale, and the
    main tradeoff, so that I can understand the harness in a few minutes.
48. As a reviewer, I want to see which configuration wins and where it costs more,
    so that the tradeoff is explicit rather than implied.

## Implementation Decisions

### Dataset

A single version-controlled dataset file holds roughly 41 questions, stratified as
~15 single-hop factual, ~10 multi-hop synthesis, ~8 entity-centric, ~8 unanswerable.
The distribution is deliberately uneven, weighted toward the strata that will
actually be tuned against.

In-corpus questions are LLM-generated from real chunks and then hand-edited to
paraphrase away literal lexical overlap with their source. This step is mandatory,
not cosmetic: questions that copy source wording inflate BM25 and make the
dense-versus-sparse comparison meaningless. A subset deliberately uses vocabulary
absent from the source passage, so that the comparison has something to discriminate
on.

A question is accepted into the dataset when a human judges that it does not hand
the retriever the source's own wording. There are two ways to violate that, and both
are the same defect:

- **Copying source vocabulary.** Measured by `containment_check.py`, which reports
  the fraction of a question's content words appearing anywhere in its source file.
- **Restating the answer in the question.** A question that asserts what it asks for
  ("which hero wrestled with Death...") smuggles the target passage's terms into the
  query even when overall containment is low.

Both inflate BM25 specifically, because sparse retrieval matches literal terms while
dense retrieval does not — so both corrupt the comparison the harness exists to make.
Leaking the answer to a *reader*, in vocabulary the source does not use, is a
different thing and is acceptable for a retrieval-only MVP; revisit it if generation
and groundedness land in Phase 2.

The containment score is **triage, not a gate**: its `>=0.50` threshold selects
questions for human review, and the human decides. Rewriting a question purely to
lower the number produces stilted phrasing that is no harder for BM25 to match. A
question above the threshold is acceptable when a human has judged it well-paraphrased
and recorded why in its `notes`.

Coverage is judged over **documents, not facts**. A question contributes a query and
a labelled set of relevant documents; which fact within a document it targets does not
affect any metric. Spread across the corpus matters, and so does spread of retrieval
difficulty; covering any particular myth or character does not.

Unanswerable questions are drawn from mythologies outside the corpus (Norse,
Egyptian, Arthurian, modern fantasy). Each must be verified absent from the corpus
before inclusion — the source text is a *comparative* mythology work that name-drops
other traditions, so absence has to be checked rather than assumed.

Relevance is graded on three levels: `0` irrelevant, `1` related context, `2`
directly answers. Three levels rather than four because a single labeller across
~41 questions will not apply finer distinctions consistently, and inconsistency
becomes noise in the metric the harness exists to make trustworthy.

Ground truth is labelled against **source documents**, not chunk identifiers. Chunk
IDs are deterministic but are a function of the chunker's size and overlap settings,
so labelling against them means any re-chunking silently rots the dataset. Chunker
parameters are recorded in the dataset as provenance.

### Judging

Candidates to grade are collected by **pooled judging**: run all configurations,
take the union of their top-10 results, deduplicate, grade that pool once. Anything
outside the pool scores zero.

This is chosen over grading only the current retriever's output because the
alternative biases the gold set toward today's system and penalises a genuinely
better future retriever for surfacing good documents nobody labelled. Pooling is
the larger upfront cost in this MVP — on the order of a few hundred unique chunks
after deduplication — and it is the cost that keeps the baseline fair to retrievers
that do not exist yet, including a future Azure backend.

Pool generation is automated; grading is manual.

### Configurations

Five arms: dense-only, sparse-only, hybrid, hybrid-plus-reranker, and
hybrid-plus-reranker with the `god_context` filter applied.

The fifth arm is an addition beyond the original brief and exists to measure a
specific suspected confound. No corpus file carries YAML frontmatter, so the god
metadata is assigned by a regex fallback that labels a chunk `unknown` whenever it
mentions more than one deity. In prose of this kind that is likely most chunks,
which would make the filter close to inert — and this arm is the production path.
The harness should report how often the filter matches, converting a suspicion into
a measurement.

### Metrics

**nDCG at 3, 5 and 10** is the headline metric. Graded relevance requires it:
Recall and MRR are binary-relevance metrics and cannot read the three-point scale.

**Recall at 3, 5 and 10** is reported as a legible secondary figure. **MRR at 10**
is reported. **Latency** is reported at p50 and p95 per configuration.

For the unanswerable stratum, the measure is precision at a confidence threshold —
the correct behaviour is returning nothing, so ordinary recall is undefined.

K values of 3, 5 and 10 are chosen deliberately: 3 is what production requests, 10
is the pooling depth, 5 sits between them and is the gating figure.

The gate is **aggregate nDCG@5**.

### Retrieval interface

A retrieval protocol with a `retrieve(query, k, filters)` shape, plus a normalising
adapter producing one uniform retrieved-item type.

This is required for the MVP, not optional and not deferred to the Azure work. The
two existing retrievers are not interchangeable: one accepts a query string, the
other accepts a pre-tokenised term list, and they return different field sets — one
returns a tidy record, the other splats the whole raw chunk. No metric can score
both arms until they are normalised.

The adapter **wraps** the existing agents without altering their internals, so the
harness cannot destabilise the working application.

Note for implementers: the caller-supplied `k` does not currently control candidate
pool sizes. Dense and sparse fan-out and the reranker's input pool are hard-coded
inside the orchestrator; `k` only sets the reranker's output size. Parameterising
those pools is **not** required for the MVP and is only needed if pool-size sweeps
are wanted later.

### Indexing

An explicit index-building step with a pinned working directory. Today the index is
created as a side effect of constructing the RAG system, at a path relative to the
current directory, which means the vector store lands wherever the process was
launched from.

The existing chunk cache invalidates on corpus file modification times only, so a
change to chunk size or overlap will not bust it. Any cache key used by the harness
or CI must include the chunker parameters explicitly.

### Fusion defect

The hybrid fusion step is **left unfixed for this spec**. The baseline is
established against the system as it currently behaves, defect included; the fix
lands in a separate follow-up change that demonstrates the metric moving.

This is deliberate. Fixing it first would forfeit the clearest available
demonstration that the harness catches real defects. The README must state this
explicitly, or a reader will reasonably assume the defect was simply missed.

### CI

CI is greenfield — there is no workflow directory and no test suite today. Ordering
matters:

1. Fix packaging so the project installs on Linux. It currently pins a
   Windows/AMD-specific torch variant and a dependency requiring a CUDA build
   toolchain, and a required NLP model's weights are not declared as a dependency.
2. Rename the existing no-assertion smoke script out of pytest's collection range.
   It downloads a multi-gigabyte model and would be collected the moment pytest runs.
3. Cache the built index, keyed on chunker parameters plus a corpus hash, with a
   scheduled cold rebuild to catch drift.
4. Add the gate.

The gate runs only on changes touching retrieval, ingestion or evaluation paths. It
is a hard failure. The override path is an explicit baseline-update commit, so that
an accepted regression is recorded as a decision rather than waved through.

Tolerance is **measured, not chosen**: run the evaluation 3–5 times to establish the
run-to-run noise floor, then set tolerance above it. Gating tighter than the noise
floor produces red builds on unrelated changes, and a gate that cries wolf gets
ignored.

Gating is on the aggregate figure only. Per-stratum results are reported as
diagnostics and explicitly marked indicative, because eight to fifteen questions per
stratum means a single question flipping moves a stratum by several points.

## Testing Decisions

A good test here asserts on external behaviour — the number a metric returns for a
given ranking, the shape the harness produces for a given set of retrieved results.
Tests should not reach into how a metric is computed internally, nor assert on the
structure of intermediate values. They must not require a model, a vector store, or
a network call.

Two seams, confirmed with the developer before writing this spec.

**Seam 1 — metric functions (primary).** The metric module is pure: ranked
identifiers plus a grade map in, a number out. No I/O, no models. Nearly all testing
lives here, because if the metric mathematics is wrong then every number the harness
produces is fiction, baseline included.

Cases to cover: nDCG against hand-computed values for a known ranking; the ideal
ranking scoring 1.0; a reversed ranking scoring materially lower; ties in grades;
an empty result list; a result list where every grade is zero; K larger than the
number of results returned; K larger than the number of relevant documents; Recall
and MRR against hand-worked examples; the unanswerable-stratum threshold check
where returning nothing is correct.

**Seam 2 — the retrieval protocol boundary.** A fake retriever implementing the
protocol lets the runner be exercised end to end — configuration matrix, pooling,
aggregation, baseline comparison, delta reporting — with no ChromaDB, no embedding
model and no reranker. This is the seam that makes the harness testable at all, and
it is the same seam a future Azure backend would slot into.

Cases to cover: the runner visits every configured arm; pooling produces the union
of arms' results and deduplicates correctly; unjudged results score zero;
baseline comparison reports the correct direction and magnitude; a regression beyond
tolerance is flagged; a change within tolerance is not.

**Deliberately not a seam: the real retrieval agents.** Testing them against a live
index would be slow, non-deterministic, and would measure the same thing the harness
already measures. The evaluation *is* the test of retrieval quality; asserting on it
in pytest as well would duplicate it in a weaker form.

**Prior art: none.** The repository has no test files, so this work establishes the
pattern. The one existing test-named file is a no-assertion smoke script and is
renamed as part of this work.

Seam 1 has no heavy dependencies and can run in a minimal environment. Seam 2
requires the package to be importable, which is gated on the packaging fix.

## Out of Scope

- **Generation and groundedness.** Explicitly deferred. Grounding measures generated
  prose against retrieved context, needs the local chat model, is non-deterministic,
  roughly doubles runtime, and pulls LLM-as-judge machinery into the gate. It is a
  strong Phase 2 and it is not this.
- **Azure AI Search adapter.** The retrieval protocol is built so this can be added
  cleanly later, but no Azure backend is implemented or evaluated here.
- **LLM-as-judge scoring of any kind.**
- **Infrastructure automation and deployment tooling.**
- **Any user interface.**
- **Fixing the hybrid fusion defect** — deliberately deferred to a follow-up so the
  fix can be measured. See Implementation Decisions.
- **Parameterising the orchestrator's hard-coded candidate pool sizes.** Only needed
  for pool-size sweeps, which the MVP does not do.
- **The embedding-model upgrade.** The harness exists partly to adjudicate it; making
  the swap is separate work.

## Further Notes

The intended sequencing is protocol and adapter, then the index build step, then
metrics, then the dataset, then the baseline, then CI — each piece testable before
the next depends on it.

The manual labelling pass is the one task that cannot be delegated. Everything else
can be built without the developer present, so the harness and the candidate pool
should be generated first, leaving labelling as a single focused sitting against a
prepared list rather than something interleaved with development.

Two consequences of decisions above are worth restating because they are implied
rather than stated: the first committed baseline will encode a known-defective
fusion step, and the labelling cost is front-loaded by pooled judging. Both are
deliberate.

Once the harness exists, the first two questions worth putting to it are the
documented embedding upgrade, whose quality claim is currently unmeasured on this
corpus, and the fusion fix.

The repository has no glossary or architecture decision records yet. Domain
conventions direct that their absence be treated as normal, and that they be created
lazily when terms or decisions actually resolve. Several terms stabilised during the
design conversation behind this spec — *arm*, *stratum*, *pooled judging*, *graded
relevance*, *the gate* — and are candidates for a glossary entry if this vocabulary
persists.
