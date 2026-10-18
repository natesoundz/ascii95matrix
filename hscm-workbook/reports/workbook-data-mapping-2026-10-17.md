# Workbook Attributes ↔ Available Data — Mapping (2026-10-17)

Purpose (Nate's direction): modify HSCM so the whole workbook serves as
L0 (replacing the original small primitive set), with everything else
derived from or connected through pointers inside HSCM, and the SQLite
lexicon database as the data side. This document maps every workbook
section to the data actually available in the `webster_full_lexicon`
export (canonical_sqlite_evidence.zip), and names the binding mechanism
or the gap for each.

Workbook sections are the 28 group headers in the sheet; consolidated
here to the 22 numbered sections (03–06E = the phonetic block).

| Section (workbook cols) | Dims | Data source in the export | Mechanism / status |
|---|---|---|---|
| 01 Character Identity (5–34) | 30 | unicode_character (codepoint, name, category); workbook itself | Mostly resolved already (K/identity). Direct. |
| 02 Math / Numeric invariants (35–47) | 13 | unicode_character.numeric_value, decomposition | Direct/derived. |
| 03–06E Phonetic / articulatory (48–150) — THE 103 | 103 | pronunciation (135,166 rows, 39 ARPABET phones, stress, syllables) joined via lexeme_id to form (spellings) | **Bridge to build** (below). Raw material present; no phone→feature table, no grapheme↔phone alignment yet. |
| 07 Pronunciation distributions (151–159) | 9 | pronunciation aggregates (phone inventory, stress patterns, syllable counts) | Derived statistics over the same join. |
| 08 Acoustics (160–186) | 27 | — | **Gap.** No acoustic measurements anywhere in the export. Needs an external source or stays deferred. |
| 09A–C Glyph geometry/topology/spatial (187–230) | 44 | unicode_character (class-level facts only) | **Partial gap.** Glyph geometry (DejaVu raster measurements used in binder run 1) is not in this export. |
| 10 Confusability (231–242) | 12 | Derived from glyph data + Unicode confusables | Derived; depends on 09 sources. |
| 11 Orthographic / positional (243–262) | 20 | form (697,611 spellings), occurrence positions | Derived statistics: letter position distributions over real spellings. |
| 12 Corpus statistics (263–282) | 20 | occurrence (254,820), collocation (274,211), construction_frequency (19,716) | Direct aggregates. |
| 13 Morphology (283–300) | 18 | inflection_form (652,092), form, occurrence_classification | Direct. |
| 14 Grammar / syntax (301–314) | 14 | occurrence_classification (193,027 warranted UD), structural_observation (1,344,889), sense.pos | Direct. |
| 15 Semantic / lexical relations (315–329) | 15 | sense (206,941), relation (731,474 typed), evidence semantic classes, FrameNet/VerbNet rows | Direct; relation rows keep their types (R cells stay relational per the registry contract). |
| 16 Punctuation / writing-system roles (330–348) | 19 | occurrence contexts + unicode_character classes | Derived from occurrence contexts. |
| 17 Programming-language roles (349–368) | 20 | — (binder run 1 used project Python source text) | **Partial gap** in this export; needs a code corpus source. |
| 18 Mathematical-syntax roles (369–390) | 22 | Occurrence contexts where available | Partial; derived, source-limited. |
| 19 Contextual / situated state (391–407) | 17 | — | **Gap.** discourse_annotation is empty (0 rows) in the source database. |
| 20 Capability distributions (408–417) | 10 | Aggregates over bound dimensions | Derived once other sections bind. |
| 21 Evidence / provenance (418–436) | 19 | source (10 sources), evidence.provenance_json | Partial/direct. |
| 22 Source classes (437–441) | 5 | source table | Direct. |

Empty source tables in the database itself (0 rows): register_frequency,
domain_assignment, discourse_annotation, primitive_assignment — any
dimension whose only warrant would come from register/domain/discourse
annotation has no evidence in this package.

## The phonetics bridge (the centerpiece)

Data path, all inside the export:
`form.normalized` (spelling) —`lexeme_id`→ `pronunciation.phones`
(ARPABET sequence) + stress_pattern + syllable_count.

Two components must be built; neither exists yet:

1. **Phone → articulatory feature decomposition.** The 39 ARPABET
   phones are a finite, closed set. Each phone decomposes over the
   workbook's own phonetic dimensions: consonants by place (bilabial,
   labiodental, dental, alveolar, postalveolar, palatal, velar, glottal),
   manner (stop, fricative, affricate, nasal, approximant, lateral),
   voicing, plus secondary/temporal properties; vowels by tongue height
   (close → open), backness, rounding, tenseness, length. This table is
   authored once against the registry's dimension IDs — it is a
   *definition*, not a measurement — and every value it asserts is
   checkable against standard phonetics.
2. **Grapheme ↔ phone alignment per lexeme.** For each lexeme with both
   a spelling and a pronunciation, align letters to phones, then
   aggregate per character: the distribution of phones each letter
   realizes, weighted by warranted occurrences, with counts kept as
   provenance. Per the registry's rules, a character/dimension cell
   binds only from observed, warranted evidence — missing stays
   missing, never zero — and relation-valued facts stay relational.

Output of the bridge: for each of the 95 characters, evidence-backed
values (or explicit unresolved states) for the 103 phonetic dimensions
and the 9 pronunciation-distribution dimensions, each cell carrying its
provenance (lexeme count, phone counts, source ids). That is the first
real binding of the 103 from data, and it exercises exactly the
connection Nate identified: workbook phonetic attributes ↔ dictionary
pronunciation ↔ spelling.

## HSCM integration shape (to be finalized against the code read)

- HSCM's original design is already pointer-based: config.py's
  DictPointer holds file pointers + a lazily built word→byte-offset
  index and never loads whole files ("Sets pointers only"). The
  workbook-as-L0 design extends the same pattern: L0 = the 437
  dimension definitions (id, name, section, interaction, eligibility,
  defer scope, per-character status) loaded from the registry; every
  higher structure references dimensions and characters by registry ID
  (pointer), never by copied values.
- The SQLite side attaches as evidence providers keyed by the mapping
  above: a resolver per section declares which tables it reads and
  which dimension IDs it can bind, mirroring the binding manifest's
  status_dispatch / required_occurrence_resolvers fields.
- Constraint mechanism: the workbook's gates (X = structurally
  impossible, eligibility, interaction types) act as constraints at
  L0 — a derived structure that would assert a value for an ineligible
  character/dimension pair is rejected at bind time, not scored down.
