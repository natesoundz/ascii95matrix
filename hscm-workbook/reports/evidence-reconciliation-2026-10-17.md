# Evidence Reconciliation — canonical_sqlite_evidence.zip vs the Workbook's Unresolved Dimensions (2026-10-17)

## The 103 physical features — identified

The 103 are a contiguous block inside the 437: **sheet columns 48–150
(registry indices 43–145), every one in the `phonetic.*` namespace**,
spanning workbook sections 03 (Phonetic / Articulatory — General)
through 06E (Phonetic Temporal Properties). First:
`phonetic.phonetic_articulatory_general.syllabic`; last:
`phonetic.phonetic_temporal_properties.geminate`. The block contains
exactly the physical articulatory attributes — vowel height and tongue
position (close, near-close, close-mid, mid, open-mid, near-open, tense,
long, ...), consonant place and manner, laryngeal/phonation, secondary
articulation, airstream, and temporal properties. It is bounded by
section 02 (Mathematical / Numeric) before it and section 07
(Pronunciation Distributions) after it. Count verified: exactly 103.
They are dimensions of the same grid, not a separate system —
consistent with Nate's clarification.

## The evidence package

`canonical_sqlite_evidence.zip` (Drive, uploaded 2026-10-16) is a
complete read-only JSONL export of `webster_full_lexicon.sqlite`
(manifest.json inside, export dated 2026-10-16 17:21): 27 tables,
~1.66 GB uncompressed. Spot integrity check passed:
pronunciation.jsonl's SHA-256 matches the manifest exactly.

Largest tables: evidence 1,682,510 rows; structural_observation
1,344,889; relation 731,474; form 697,611; inflection_form 652,092;
lexeme 590,292; occurrence 254,820; collocation 274,211; sense 206,941;
occurrence_classification 193,027; pronunciation 135,166;
webster_entry / webster_lexeme 86,036 each (the dictionary package the
status document names); unicode_character 40,575.

Evidence-table categories: INFLECTED_FORM 652,092; POS 479,328;
NOUN_SEMANTIC_CLASS 146,312; PRONUNCIATION 135,166; CONCRETENESS 79,908;
DOMINANT_POS_NORM 39,954; PRONTYPE 36,213; ADJECTIVE_CLASS 30,002;
VERB_SEMANTIC_CLASS 25,047; FRAMENET_FRAME 13,572; COUNTABILITY 11,869;
UNICODE_CLASS 9,473; VERBNET_CLASS 9,022; plus smaller classes —
1,809 distinct (category, subcategory) pairs.

Four tables are EMPTY in the source database itself (0 rows):
register_frequency, domain_assignment, discourse_annotation,
primitive_assignment.

## Reconciliation against the unresolved dimensions

Baseline (from the repo's binding_manifest.json, which mirrors the
workbook exactly): 413 of 437 dimensions carry at least one
unresolved-status cell (E/M/R/S/U/V); all 103 phonetic dimensions are
among them. (The status document's "212 bound / 225 unresolved" is the
first binder run's evidence count; the per-cell resolver view is the
stricter one used here.)

- **Morphology (18), grammar (14), semantic (15)** — strong raw
  coverage: inflection forms, 193k warranted UD occurrence
  classifications, 479k POS evidence rows, 1.34M real-UD structural
  observations, WordNet-style relations (731k), FrameNet/VerbNet
  classes, semantic classes, concreteness, countability.
- **Pronunciation (9) and the phonetic 103** — raw material present,
  resolver not built. 135,166 pronunciations over 39 ARPABET phones
  with stress and syllable counts. But pronunciations attach to
  lexemes, not characters, and the export contains no
  phone-to-articulatory-feature table. Binding the 103 requires
  (a) a feature decomposition of the 39 phones (place, manner,
  voicing, tongue height/backness, etc.) and (b) grapheme-to-phone
  alignment per lexeme to project phone features onto letters.
  Neither exists yet. Data-quality note: a small number of
  pronunciation rows carry non-phone tokens in the phones field
  ('danish', 'french', 'name,' ...) — a cleaning rule is needed.
- **Acoustics (27)** — no acoustic measurements in this package (no
  formant/audio tables). Remains unresolved from this source.
- **Glyph (44)** — partial: unicode_character supplies identity/class
  facts for all codepoints, but glyph geometry (the DejaVu raster
  measurements used by the first binder) is not in this export.
- **Corpus statistics (20)** — good raw coverage: collocations (274k),
  construction frequencies (19.7k), occurrences (254k).
- **Context (17), capability (10), and the register/domain/discourse
  dependent dimensions** — NO evidence in this package: the four
  register/domain/discourse/primitive tables are empty in the source
  database itself.

Bottom line: this export is the lexicon-side evidence the status
document said was needed, and it covers the lexical, morphological,
grammatical, semantic, and corpus blocks in bulk. It does not by
itself bind the phonetic 103 (needs the phone-feature bridge), the
acoustic block (no data), or the register/discourse-dependent
dimensions (empty source tables).

## HSC / HSMC in Drive — status

- Drive's only HSC-named folder is `HSCM` (April 2026 snapshot:
  manifold_engine.cpp, engine.py, matrix.py, orchestrator.py,
  config.py, demo.py).
- Newest HSCM-named artifacts: HSCM_Coordinate_IR_Final.zip and
  HSCM_Computational_v1.zip (Oct 3, 2026). Nothing HSC-named was
  modified Oct 15–16; the folder modified Oct 15 ("Prove this and
  invert the ml landscape") is empty in Drive, and the only file
  modified Oct 16 is canonical_sqlite_evidence.zip itself.
- So the most recent augmentation visible in Drive is this evidence
  export (Oct 16). If a newer HSMC working state exists on the local
  machine, it has not reached Drive yet.
