# Real Layer-One Dataset and Artifact Build

This note defines the real-data path from the canonical 437-column registry to
the first frozen target-free transformer layer.

## Inputs

1. `data/Canonical_ASCII95_Attribute_Matrix_v1.xlsx`
2. `data/registry_contract.json`
3. `data/binding_manifest.json`
4. `webster_full_lexicon.sqlite`
5. real local text/code corpora supplied to `tools/assemble_real_dataset.py`

The lexical database is used because it already preserves real source evidence:
UD English EWT occurrence parses, WordNet, VerbNet, FrameNet, BECL, Unicode,
CMUdict, UniMorph, lexical/sense relations, pronunciation, morphology,
structural nodes/edges, discourse annotations where present, collocations,
register/domain evidence, errors/violations where present, and provenance.

No generated sentence is accepted as verification evidence.

## Dataset products

`tools/assemble_real_dataset.py` writes:

- `records.jsonl`: literal retained ASCII95 text/code plus occurrence annotations
- `lexical_snapshot.jsonl`: heavy lexical/sense evidence once per observed lexeme
- `rejections.jsonl`: every excluded record and reason
- `manifest.json`: source hashes, counts and invariants

The assembler never converts unsupported characters to ASCII. It rejects the
record instead, preserving the source in the rejection audit.

## Canonical dimension source families

| Registry family | Primary real evidence |
| --- | --- |
| character_identity / numeric | canonical workbook + Unicode data + deterministic numeric facts |
| phonetic / pronunciation | CMUdict / pronunciation tables plus an explicitly sourced phone-feature table |
| acoustic | measured speech/acoustic source; no substitute |
| glyph / confusability | actual rendered-font ensemble measurements; no substitute |
| orthographic | literal occurrence text and declared span boundaries |
| corpus | statistics over the exact frozen dataset snapshot |
| morphology | UD morphology, UniMorph, form/lemma/inflection evidence |
| grammar | UD token/dependency evidence + structural nodes/edges |
| semantic | Webster, WordNet, FrameNet, VerbNet and typed lexical relations |
| punctuation | occurrence structure/corpus evidence |
| programming | Python tokenize/AST for Python; explicit grammar source for other languages |
| math_syntax | explicitly parsed mathematical notation/context |
| context | active occurrence annotations produced by the corresponding families |
| capability | distributions derived from warranted observations |
| evidence / source_class | source/provenance and coverage statistics |

## Important distinction: annotations versus scalar measurements

The 437-column registry contains booleans, values, deferred categories, typed
relations, measurements, modulators and structural/derived fields.

The current `relational_compiler.py` consumes one scalar `g[i,k]` per
occurrence/dimension. Therefore a typed fact may enter `g` only after its
projection law is declared.

Examples:

- a fixed workbook `1` or `0` has an immediate scalar interpretation;
- `orthographic.word_length` has an observable scalar once a word span is
  warranted;
- `semantic.hypernym_relation` is a typed graph relation and must not be
  converted to an arbitrary integer just to fill one scalar;
- `character_identity.unicode_general_category` is categorical and requires a
  declared projection/constraint law rather than category-number assignment;
- `glyph.geometry.bounding_width` requires an actual rendering measurement.

Missing, ineligible and unresolved are not zero.

## Strict build gate

A full 437-dimensional artifact may be published only when every dimension that
requires occurrence evidence has:

1. an explicit source,
2. an explicit scope,
3. an explicit value or typed-relation representation,
4. an explicit deterministic projection/constraint law,
5. provenance traceability,
6. nonzero valid coverage where the dimension is expected to occur.

If a dimension fails that gate, the build report must identify it. Do not fill
it with a neutral constant merely to make `H[k] > 0`.

## Layer-one compile order

Once the binding gate passes:

```text
FROZEN DATASET SNAPSHOT
        |
        v
TYPED OCCURRENCE EVIDENCE
        |
        v
DECLARED DIMENSION PROJECTIONS / CONSTRAINTS
        |
        v
COMPILE E[95,437]
        |
        v
VERIFY + FREEZE E
        |
        v
DERIVE CONTEXTUAL RESIDUAL REQUIREMENTS
        |
        v
COMPILE TARGET-FREE ATTENTION
        |
        v
VERIFY + FREEZE ATTENTION
        |
        v
REPLAY FROZEN E + ATTENTION
        |
        v
COMPILE REMAINING FFN RESIDUAL
        |
        v
VERIFY + FREEZE FFN
        |
        v
TARGET-FREE READOUT
        |
        v
WINNER / SURVIVORS / ELIMINATED
```

Ground-truth target `y` is permitted while compiling requirements. It is
forbidden as a runtime input.

## Required real-artifact report

The final layer-one report must include:

- input dataset hashes
- registry/workbook hash
- total records and character occurrences
- annotation coverage by source family
- binding coverage for every one of 437 dimensions
- contradictions/unresolved constraints
- `E[95,437]` verification
- embedding residual
- post-attention residual
- post-FFN residual
- exact serialized tensor checks
- target-free replay results
- winner/survivor/eliminated records
- total compiled parameter allocation
- explicit `runtime_target_input: false`

## Layer-two gate

Do not compile layer two merely because code for a second block exists.

Layer two begins only after layer one is a real, frozen, target-free artifact.

Its input evidence is the state and unresolved constraint field that layer one
actually produces. Layer two must not silently revisit the raw target to repair
layer-one failures.
