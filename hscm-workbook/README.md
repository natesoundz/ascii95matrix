# HSCM × Workbook — Complete Working Set (v1, 2026-10-17)

The system as it stands after the 2026-10-17 work: HSCM's original
setup, the workbook loaded as its L0, the phonetic bridge, and the
SQLite-classification replacement for the POS heuristic stub.

## Design principles in force (Nate's)

- **The workbook is an explicit token embedding**: 95 characters ×
  437 named dimensions. Cells may hold numbers (a vector), symbolic
  states (the status codes), or both — the registry supports all
  three fillings.
- **The whole workbook is L0**, replacing HSCM's original 65-primitive
  entry layer. Everything above L0 connects by pointer (canonical
  dimension id / lexeme id), never by copied value.
- **Differentiation is tiered**: the 103 physical/phonetic attributes
  decide first; where they cannot distinguish a case, the remaining
  workbook columns differentiate — part-of-speech category and the
  other dimension groups, fed by the SQL classification tables.
- **Registry gates are hard constraints**: X = structurally
  impossible, binding forbidden; eligibility is not activation;
  missing evidence stays missing, never zero.

## Layout

- `originals/` — Nate's HSCM folder set, byte-for-byte untouched
  (config.py, matrix.py, engine.py, orchestrator.py, demo.py,
  manifold_engine.cpp, hscm_full.py).
- `src/workbook_l0.py` — the workbook as L0: all 437 dimensions with
  gates as a constraint API; sparse per-character pointer views.
- `src/phonetic_bridge.py` — 39 ARPABET phones decomposed into
  workbook phonetic dimension ids; spelling↔pronunciation join;
  positional alignment on the 1:1 subset; tiered binding candidates
  with provenance. Results in `results/bridge_output.json`
  (240 candidate cells, 158 strong, 0 gate violations).
- `src/build_classifier_index.py` + `src/lexicon_l1.py` — the stub
  replacement: per-lemma classification index aggregated from the
  SQLite tables (senses, semantic classes, countability, concreteness,
  POS evidence, warranted UD classifications, form inheritance), and
  the evidence-driven (primitive, subpos) classifier with the stub's
  exact call contract and 100% edge validity against matrix.py.
- `registry/` — registry_dimensions.tsv + binding_manifest.json
  (exported from the canonical workbook; identical content to the
  workbook, verified cell-by-cell).
- `reports/` — the five audit/build reports from 2026-10-17.

## Running it

1. Data: the JSONL table extracts come from
   `canonical_sqlite_evidence.zip` (Drive, 2026-10-16 — a full export
   of webster_full_lexicon.sqlite). Extract lexeme/form/pronunciation/
   sense/occurrence/occurrence_classification JSONL next to the code
   in a `data/` folder.
2. `workbook_l0.py`: set `REGISTRY_DIR` (top of file) to the folder
   holding registry_dimensions.tsv + binding_manifest.json (this
   package's `registry/` works).
3. `python build_classifier_index.py` regenerates
   `data/classifier_index.json` (~40 s; the generated file is ~74 MB
   and is therefore not shipped in this package).
4. `python phonetic_bridge.py` and `python lexicon_l1.py` run their
   verifications end-to-end.
5. Orchestrator integration: instantiate `LexiconL1()` once at
   startup; the body of `_assign_primitive_subpos` becomes a lookup
   into it. Nothing else in orchestrator.py or the engine changes.

## Known open items (from the reports)

- `hscm_sparse.cpp` ("HSCM Pure") is not in Drive or any zip — the
  sparse engine itself is still needed from Nate's machine.
- matrix.py's edge table leaves 10 of 45 subpos unreachable
  (~19k degraded assignments, sized per class in the classifier
  report); extending it is a one-line-each decision.
- C++ `TrajSet` is uint8_t but trajectory masks need 10 bits —
  axis-4 poles silently truncate in the C++ manifold.
- The lexicon's CMU stratum (lexeme ids 86,037+) has no classification
  evidence; ~74k words fall back, flagged.
- Next build step per the tiered-differentiation principle: wire the
  non-phonetic workbook columns as explicit tie-breakers in the
  classifier/manifold layer (POS category already participates via
  the classification tables).
