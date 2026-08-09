# Grading Rubric — Ticket #10 (Manual Relevance Grading)

Prep material for the manual grading sitting. Turns the three-point scale in
`spec.md` / `myth_eval/dataset.py` into a checklist a single labeller can apply
consistently across a few hundred pooled candidates without re-deriving intent
each time.

## The decision tree

Walk in order — **2, then 1, then 0.** Stop at the first grade whose
conditions are met. Grade against the source document as a whole (skim
neighbouring chunks from the same document if the pooled chunk cuts off
mid-fact) — labels key on `source_document`, not chunk ID, so a document
gets exactly one grade regardless of how many of its chunks are in the pool.

Grade candidates **independently**. Don't compare a candidate against other
candidates in the pool before assigning it a grade — each document either
contains the queried fact or it doesn't, on its own.

| Grade | Assign when |
|---|---|
| **2 — directly answers** | The document contains a sentence that states the specific fact the question's answer-slot asks for — under any vocabulary, including a Greek/Roman name variant (Cupid/Eros, Diana/Artemis, Minerva/Athena, Neptune/Poseidon, Venus/Aphrodite, Jupiter/Zeus, Juno/Hera). For a multi-hop question, this applies per hop: a document fully answering one hop of a compound question is a 2, independent of whether other hops are covered elsewhere. |
| **1 — related context** | Same entity or myth as the question, but the document does not state the queried fact — background, setup, consequence, or an adjacent scene from the same character's arc. |
| **0 — irrelevant** | Neither of the above: wrong entity, unrelated myth, or thematic-only overlap with no shared fact. **Unanswerable-stratum questions are not graded at all** — see below. |

**When in doubt, default down a grade rather than up.** That's the rule doing
the most work for consistency across a solo labelling pass — every other
line in this doc exists to reduce how often you need it.

---

## Grade 2, worked

`factual-03` — "Which tiny labourers pitied Psyche and finished dividing the
jumbled pile of grain Venus had set before her?"
Candidate: `023_CUPID_AND_PSYCHE.md` (the passage naming the ants that took
pity on Psyche and sorted the grain heap).
→ **2**. Contains the specific fact the answer-slot asks for.

`synthesis-03` — "...one attempt fails because the rescuer looks back too
soon, the other succeeds because a visiting hero physically wrestles the
spouse's life away from Death himself. Identify the couples in each case..."
Candidate: `068_ADMETUS_AND_ALCESTIS.md` (the passage where Hercules wrestles
Death for Alcestis).
→ **2**. Fully answers the "wrestles Death" hop by name (Admetus/Alcestis,
Hercules), on its own terms. It says nothing about Orpheus/Eurydice — that
doesn't matter; that's a separate document, graded independently. (For a
2-hop question with two documents each graded 2, `ndcg_at_k`'s ideal ranking
is `[2, 2]` — an arm that surfaces both near the top scores close to 1.0,
which is the intended reward for solving a synthesis question.)

## Grade 1, worked

`entity-02` — "Who equipped Perseus with the buckler and footwear enabling
him to withstand Medusa?"
Candidate: `034_PERSEUS_AND_ATLAS.md` — Perseus, golden apples, Atlas
refusing him hospitality after the Gorgon's slaying.
→ **1**. Same entity (Perseus), a real episode in his arc, but this document
never says who armed him — that fact is stated in `033_PERSEUS_AND_MEDUSA.md`
("the former of whom lent him her shield and the latter his winged shoes"),
which is graded 2 in its own right. Two documents, two independent grades:
`034` doesn't inherit relevance from `033` just because both mention Perseus.

## Grade 0, worked

`unanswerable-08`'s top dense hit, `067_ACHELOUS_AND_HERCULES.md` (score
0.6156), is a genuine near-miss (shapeshifting river-god, serpent form,
defeated by a hero) already investigated and recorded absent in that
question's `notes`. If it surfaces as a pooled candidate for a *different,
answerable* question with no shared entity or fact, it's a 0 — thematic
proximity (serpent, hero, combat) without a shared fact or entity is
irrelevant, not related context. Don't let "this document keeps coming up"
read as a signal of relevance.

---

## Unanswerable-stratum questions: do not grade

`dataset.py` documents `relevance` as **empty by construction** for
`unanswerable-*` questions — the empty map *is* the ground truth, because
`Question.is_unanswerable` routes these straight to `unanswerable_precision`
and the grade map is never consulted. `is_labelled` returns `True` for them
precisely because there is nothing to label.

**Skip these questions entirely during grading.** Do not write explicit 0s
into their `relevance` map — a populated map reads to a future maintainer as
"grading happened here," when what actually happened is "grading doesn't
apply here." If a candidate pool generator (#9) still surfaces documents
against an unanswerable question, that's a diagnostic about pooling, not a
grading task.

---

## Explicitly not part of this rubric

- **Chunk quality.** You're using one chunk as a window into whether its
  *source document* answers the question — not judging the chunk's writing
  or completeness.
- **Retriever score or rank.** Grading is blind to which arm(s) surfaced a
  candidate or how they ranked it. Grading by score would bias the gold set
  toward whichever arm generated the labels — the exact bias pooled judging
  exists to avoid (spec.md, "Judging").
- **Thematic vibes.** Sharing a mood, setting, or motif without sharing a
  fact or entity is grade 0. This is the most common place inconsistency
  creeps in — see the Grade 0 worked example above.

## Recording disagreement with yourself

If a candidate takes real deliberation to place, write one line in that
question's `notes` in `questions.json` — the same way `unanswerable-*`
questions already record their verification reasoning. This doc is
calibration for the common cases; a note on the hard ones is what lets a
second labeller, or future you, understand why a boundary call landed where
it did without re-deriving it from scratch.
