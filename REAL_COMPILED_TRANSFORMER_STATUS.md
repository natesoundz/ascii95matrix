# Real Compiled Transformer Build Status

This file records the first **real-data** canonical ASCII95 compile and the start of Layer 2.

It is not a synthetic integrity test.

## 1. Canonical source

The authoritative registry remains:

- 95 printable ASCII characters
- 437 typed relationship dimensions
- `data/Canonical_ASCII95_Attribute_Matrix_v1.xlsx`
- workbook SHA-256: `065cb5424ec0fd2d206de093e13ed2317f29a0d6a1eda837df1a9d46f81506ab`

The registry defines measurable relationships, eligibility, gates, defer scopes, and interaction semantics. It does not directly prescribe embedding coordinates.

## 2. Real annotation/evidence arena

The first real binding run used:

- 2,601 real records
- 144,413 real character occurrences
- real Webster headwords and definitions from the project's 86,036-entry dictionary package
- actual project Python/source text
- all 95 printable ASCII characters represented
- deterministic DejaVu Sans Mono glyph raster measurements

The binder generated:

- 16,186 finite aggregate character/dimension measurements
- full typed visual/corpus relations retained separately instead of being silently flattened
- exact source hashes and a per-dimension binding report

### Current coverage

- 212 / 437 dimensions have real bound evidence
- 177 / 437 are geometrically active with nonzero spread
- 35 bound dimensions are degenerate under the current evidence and remain neutral
- 225 dimensions remain unresolved
- active embedding rank: 95

The unresolved dimensions are **not** filled with guessed values.

The largest unresolved blocks are phonetic/articulatory, pronunciation, acoustics, and lexical/morphology/grammar/semantic fields that require additional real annotation sources.

The local `webster_full_lexicon.sqlite` is the intended next enrichment source. Its existing schema can provide real UD POS/morph/dependency occurrences, occurrence classifications, lexical senses and relations, WordNet/VerbNet/FrameNet evidence, pronunciation, inflection, structure nodes/edges, discourse, collocation, register/domain, and source provenance.

## 3. Missing evidence rule

The real compiler is coverage-aware.

Missing evidence must never silently become a negative observation.

For an unobserved or structurally ineligible character/dimension cell:

```text
E[c,k] = 0   # neutral / inactive
```

and an explicit mask records why it is inactive.

Only observed, warranted evidence participates in centering and spread calculations.

The frozen geometry is still exactly:

```text
E[95,437]
```

so later evidence can activate currently unresolved dimensions without changing the architecture interface.

## 4. Frozen real embedding

The real frozen geometry passed verification:

- shape: `95 x 437`
- active dimensions: 177
- rank: 95
- maximum absolute E recomputation error: `0.0`
- neutral missing-cell check: PASS
- neutral ineligible-cell check: PASS

Compile order remains:

```text
real evidence
  -> G / whole-field statistics
  -> E[95,437]
  -> verify
  -> freeze
```

## 5. Real Layer 1 compile

The Layer-1 replay compile used a smaller deterministic real set so the complete attention + FFN solve could be finished and verified immediately:

- 75 real records
- 3,949 real character occurrences
- all 95 printable ASCII characters represented

Layer 1 remained target-free at runtime.

### Residual

```text
h0  embedding/context field      0.4978410430
hA1 post-attention-1             0.4516247928
h1  post-FFN-1                   0.4370190207
```

### Replay top-1

```text
h0   28.3616%
hA1  40.6938%
h1   41.6814%
```

### Layer-1 allocation

```text
embedding E              95 x 437        41,515
mutual relation field    14 x 95 x 95   126,350
attention-1 values       14 x 95 x 437  581,210
FFN-1 correction field   95 x 437        41,515
-------------------------------------------------
Layer-1 core total                       790,590
```

No target `y` is supplied to the runtime.

## 6. Layer 2 started and compiled

Layer 1 is frozen before Layer 2 begins.

Layer 2 is currently defined by the residual left after Layer 1:

```text
h1
  -> compile attention-2 against E[y] - h1
  -> hA2 = h1 + delta_A2
  -> freeze attention-2
  -> compile FFN-2 against E[y] - hA2
  -> h2 = hA2 + delta_F2
  -> freeze
```

The ground-truth target is available only to the compiler/replay verifier, never to target-free runtime inference.

### Layer-2 real result

```text
enter Layer 2             0.4370190207
post-attention-2          0.4334151777
post-FFN-2                0.4299331521
```

Replay top-1:

```text
h1   41.6814%
hA2  42.2892%
h2   42.8716%
```

Layer 2 therefore reduced the remaining real replay residual and improved the same target-free readout.

### Layer-2 allocation

Layer 2 shares the frozen embedding and mutual relation field.

It adds only:

```text
attention-2 values       14 x 95 x 437  581,210
FFN-2 correction field   95 x 437        41,515
-------------------------------------------------
Layer-2 added values                     622,725
```

Two-layer core runtime total:

```text
1,413,315 compiled values
```

## 7. Next full build

The next run should use the local maximal lexical database:

```text
C:\Users\idz9m\Downloads\webster_full_lexicon.sqlite
```

The required sequence is:

1. assemble provenance-preserving real records and lexical snapshots;
2. bind only warranted annotations to exact canonical registry IDs;
3. preserve relation-valued facts separately;
4. report per-dimension coverage and unresolved dimensions;
5. recompile the 95 x 437 embedding;
6. verify and freeze;
7. recompile Layer-1 attention from its remaining residual;
8. freeze Layer-1 attention;
9. recompile Layer-1 FFN;
10. freeze all Layer 1;
11. compile Layer 2 from only the residual still remaining;
12. evaluate target-free runtime separately from compilation replay.

## 8. Non-negotiable invariants

- compilation, not conventional training
- no optimizer or gradient update loop
- no random semantic initialization
- no runtime access to the unknown target
- missing evidence remains missing/neutral
- contradictions and unresolved dimensions remain visible
- relation identity is preserved when scalar projection would destroy it
- every stage is verified and frozen before the next stage is compiled
