# HSCM: POS Heuristic Stub Replaced by the SQLite Classification Tables (2026-10-17)

Nate's direction, executed: `orchestrator.py`'s `_assign_primitive_subpos()`
(word-length/phoneme heuristic) is replaced by an evidence-driven
classifier built from the lexicon database's classification tables.

## What was built (working copy, originals untouched)

`hidden_files/hscm_workbook/build/`:
- **build_classifier_index.py** — aggregates the designated tables into
  `classifier_index.json` (220,236 lemmas): sense POS + WordNet lexical
  classes (147,806 lemmas), DOMINANT_POS_NORM, NOUN/VERB_SEMANTIC_CLASS,
  COUNTABILITY, CONCRETENESS, ADJECTIVE/ADVERB_CLASS, POS evidence
  (163,428 lexemes), real UD occurrence POS + warranted proper-noun
  classifications, and **form inheritance**: a word that exists as a
  form of a classified lexeme inherits its evidence, marked as such
  (49,538 inherited entries).
- **lexicon_l1.py** — `LexiconL1.assign(word, phonemes)`, the exact
  contract of the stub (returns `(primitive, subpos)` or None when
  phonemes is falsy). Deterministic rule ladders read the index;
  every emitted pair is asserted against matrix.py's own
  VALID_L0_L1_EDGES (imported from the original file) — validity by
  construction. `assign_with_provenance()` returns the rule that fired
  and its evidence flags.
- Integration into orchestrator.py is a one-function swap: instantiate
  `LexiconL1()` once at startup; the body of
  `_assign_primitive_subpos` becomes a lookup. Nothing else changes —
  the stdin record format and the engine are untouched.

## Verification (the orchestrator's real population: 126,052 words with pronunciations)

- The heuristic stub emits exactly **3 distinct pairs** for the entire
  dictionary: (THING, NOUN_CONCRETE) for 89.5% of words, (THINK,
  VERB_MENTAL), (MOVE, VERB_MOTION). The classifier emits **44
  distinct pairs**; **64.3% of assignments change**.
- Evidence-clean assignments: 33,048 (26.2%). Degraded (evidence
  present, desired subpos unreachable — see below): 18,737 (14.9%).
  Fallback to the old heuristic, flagged: 74,267 (58.9%) — words for
  which the database holds no classification evidence at all.
- Edge validity: 100% (asserted per call; zero violations).
- Spot checks: dog → (SOMEBODY, NOUN_ANIMATE); run → (MOVE,
  VERB_MOTION); say → (SAY, VERB_SPEECH); think → (THINK, VERB_MENTAL);
  idea → (KNOW, NOUN_ABSTRACT); freedom → (KNOW, NOUN_STATE); water →
  (THING, NOUN_CONCRETE) via noun.substance (an early absolute-count
  proper-noun rule misfired on sentence-initial "Water"; fixed to
  share-based warrants and retested).

## Two findings

1. **The lexicon has two strata.** Webster/WordNet lexemes (ids
   1–86,036) are richly classified; the CMU pronunciation block
   (ids 86,037+) has pronunciations but no senses, evidence, or
   occurrences. Words whose spelling also exists in the classified
   stratum — directly or as an inflected form — classify cleanly;
   the remaining ~74k CMU-only words (names, contractions, variants)
   have no classification evidence in the database. That is a
   database coverage fact, and the classifier flags rather than
   papers over it.
2. **matrix.py's edge table is now the binding constraint.** 10 of
   the 45 subpos have NO incoming edges in VALID_L0_L1_EDGES, so no
   word can ever receive them: NOUN_PROPER (~2,900 words degraded),
   NOUN_MASS (~1,100), NOUN_COLLECTIVE (~480), NOUN_ROLE,
   ADJ_RELATIONAL (~1,200), ADJ_TEMPORAL, ADV_MANNER (~1,600 — the
   most common adverb class), ADV_FREQUENCY, PRONOUN_RELATIVE,
   DETERMINER_DEF. Separately, adjectives default coarsely
   (~6,300) because ADJECTIVE_CLASS evidence only distinguishes
   adj.all / adj.pert / adj.ppl. Extending the edge table (e.g.
   DO → ADV_MANNER, THING → NOUN_MASS) is a matrix.py decision —
   Nate's — and each added edge converts its degraded count to
   clean assignments wholesale.
