# ASCII95 Matrix

The canonical character registry is [Canonical_ASCII95_Attribute_Matrix_v1.xlsx](data/Canonical_ASCII95_Attribute_Matrix_v1.xlsx).
It contains **95 printable characters × 437 typed attribute columns**. The workbook is preserved byte for byte.

This repository also contains one executable compiler extracted from the latest active Compiler Lab branch. It produces numerical weights for that branch's **historical binary 254-coordinate architecture**. The complete **canonical 437-column registry → occurrence bindings → target-free model weights** path remains unimplemented. The historical control is explicitly isolated under `legacy/`.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| `data/Canonical_ASCII95_Attribute_Matrix_v1.xlsx` | Authoritative character registry, interaction semantics, participation gates, and legend. |
| `compile_weights.py` | One executable: validate inputs, compile evidence and staged weights, verify serialized arrays. Its `inspect` command performs workbook preflight. |
| `legacy/architecture.json` | Explicit declaration of the supported historical architecture, numerical precision, and matrix checksum. |
| `legacy/ASCII95_254_BINARY_MATRIX.tsv` | Required historical control substrate. It has no authority over the new registry. |
| `requirements.txt` | NumPy dependency. The workbook reader uses Python's standard library. |
| `verification.json` | Actual verification inputs, counts, numerical errors, residuals, and limits. |
| `LICENSE.md` | Original research-only license and attribution. |

The compiler has no project-local Python dependencies. Dataset downloaders, unrelated experiments, alternative compilers, and the target-conditioned runtime were omitted. A dataset is an explicit local input. The supported architecture is declared and validated; arbitrary architectures are not silently accepted.

## Canonical registry contract

The worksheet's status cells describe what may participate and what information must be supplied. Their meanings are preserved:

| Cell | Meaning | Compilation consequence |
| --- | --- | --- |
| `1`, `0` | Fixed positive / fixed negative | Defined boolean evidence. |
| `X` | Structurally ineligible | Preserve the gate. Do not substitute a negative observation. |
| `E` | Eligible, deferred | Bind the declared occurrence, context, pronunciation, or other source axis. Eligibility alone is not activation. |
| `R` | Relation | Retain its participants and relation type until an explicit projection is defined. |
| `M` | Measurement | Supply the realization axis, units, and deterministic transform. |
| `U` | Modulator or statistic | Modulate evidence as declared. The legend forbids treating this as a primitive coordinate. |
| `V` | Scalar/category slot | Supply its actual value and a declared encoding. |
| `K` | Identity key | Identify the row. The legend forbids using it as a weight input. |
| `S` | Structural or derived | Resolve from its dependencies, with no independent free coordinate. |

Missing or undefined values remain distinct from zero. The intrinsic Unicode math set stays **`+<=>^|~`**, separate from contextual mathematical syntax. The registry contains 380 key cells and 6,586 modulator/statistic cells, so assigning every column an independent weight would already violate its legend.

Word forms retain the same character substrate across uses. Sense, POS, WordNet relation type, and observed grammatical usage belong to contextual evidence. Taxonomy, derivation, domain, and part/whole relations must retain their separate types; this extraction does not create a WordNet compiler or collapse those relations to one distance.

## Exact historical derivation

These are the equations implemented by the extracted source. They describe what the code actually computes.

Let `A[y,k]` be the historical binary attribute for character `y`, coordinate `k`. There are 95 characters and 254 coordinates. Every supplied ground-truth occurrence is used. Objective masks control context visibility.

1. **Shared evidence counts.** Count `C[o,c,y]`: context character `c`, target `y`, relative offset `o`. Preserve the multiplicity of causal histories 1–7, radius views 1–7, legal width-four hole views, and full permitted context. Offsets cover −7…−1 and +1…+7. The farther part of full context is retained in `C_full[c,y]`. View and objective counts are stored separately as audit arrays.

2. **Conditional attribute profiles.** For each observed context:

   `p[o,c,k] = sum_y C[o,c,y] * A[y,k] / sum_y C[o,c,y]`.

   The same calculation applies to the farther full-context counts. Unsupported profiles do not contribute observed evidence.

3. **All-candidate contrasts.** A candidate's profile score is:

   `score(y,p) = 1 - mean_k(abs(A[y,k] - p[k]))`.

   Compare each observed target with all 94 other characters. Ties count as one half. The resulting pairwise win rate supplies `affordance[y,k]`, averaged over competitors whose historical bit differs at coordinate `k`. With no differing competitor, the historical equation uses 1.

4. **One shared embedding table.** Average contextual profiles where each target actually occurs, weighted by their counts, to obtain `support[y,k]`. Unsupported target rows use the historical neutral support 0.5. Set:

   `side = support if A[y,k] == 1 else 1 - support`

   `E[y,k] = (2*A[y,k]-1) * max(1e-6, side * affordance[y,k])`.

   The compiler requires rank 95. This is a prescribed support/affordance construction. The source does not implement the full global constraint-satisfaction solve described in the ground-truth-arena design.

5. **Recognition of the compiled rows.** Set `Wd = pinv(E)` and verify `E @ Wd = I95` within tolerance. This inverse recognizes the compiled embedding rows. It supplies no contextual semantic prediction by itself.

6. **Forward/reverse specificity and attention.** Compute:

   `r[o,c,y] = sqrt(P(y | o,c) * P(c | o,y))`.

   `WQ = I95`; `WK[o,c,y] = log(r[o,c,y])` wherever the relation is observed. A separate validity mask excludes unobserved relations. Transform a profile into the historical signed coordinate system with the step-4 sign, affordance, and floor. For each `(o,c)`, `WV[o,c]` is the weighted mean of `required_profile_state[y] - E[y]`, with weights `C[o,c,y] * r[o,c,y]`.

   Freeze those matrices. For a supplied target occurrence, normalize relevance across visible offsets and compute:

   `hA = E[y] + attention_delta`.

7. **Remaining FFN residual.** Replay the same frozen evidence. Let `t[i,k]` be the contextual required state derived from the relevance-weighted profiles, `d[i,k] = E[y,k] - hA[i,k]`, and `rF[i,k] = t[i,k] - hA[i,k]`. The coordinate correction is the direct normal-equation solution:

   `corr[y,k] = sum_i(d[i,k] * rF[i,k]) / sum_i(d[i,k]^2)`.

   The implementation uses zero when the denominator is at most `1e-18`. Paired ReLU units realize the signed correction:

   `delta_FFN = corr[y] * (E[y] - hA)`

   `hF = hA + delta_FFN`.

   The fixed FFN has 48,260 hidden units. Its matrix shapes are `W1[603,48260]` and `W2[48260,254]`. It uses a known-character gate and two ReLU units per character/coordinate. No gradient updates or random initialization are used.

The historical code's contextual requirement and attention both receive the target character `y`. Its original runtime reads `text[position]` before scoring. These calculations therefore verify a **known-target transformation**. They do not demonstrate prediction of an unknown next character. A target-free inference equation and its corresponding compilation obligations are still required.

## Using the extracted compiler

Requires Python 3.10+ and NumPy. Install the dependency with `python -m pip install -r requirements.txt`.

Inspect the canonical workbook without changing it:

```powershell
python .\compile_weights.py inspect .\data\Canonical_ASCII95_Attribute_Matrix_v1.xlsx
```

Supply one JSON object per line in your evidence file. Required fields are `id`, `dataset`, `objective`, and `text`. For example, this text is an actual source-code line from the audited profiler:

```json
{"id":"profiler-function","dataset":"source-code","objective":"CODE_CAUSAL","text":"def ground_truth_positions(record: dict) -> list[int]:"}
```

Objectives are `LM_HOLE`, `SFT_RESPONSE`, `CHAT_ASSISTANT`, `PROBLEM_SOLVING`, `CODE_CAUSAL`, and `CODE_INSTRUCTION`. `LM_HOLE` permits context on both sides while hiding the target position. The other objectives use preceding context. You may supply any subset of objectives; empty objectives are reported as empty.

Optional fields are `roles`, `ground_truth_positions`, and `provenance`. Positions default to every character and must otherwise be sorted, unique integers inside the record. Context-only copies must be explicitly excluded from truth positions. Each `(dataset,id)` must be unique. Text must already be printable ASCII95. There is no implicit Unicode removal, newline conversion, or fabricated annotation. Record boundaries must be declared by the dataset preparation process.

Compile the historical control against your local evidence:

```powershell
python .\compile_weights.py compile --dataset .\evidence.jsonl --architecture .\legacy\architecture.json --output .\compiled_output
```

The output directory must be empty. The compiler freezes a literal `evidence.jsonl` snapshot and produces `compile_report.json`. It publishes `weights.npz` only after numerical verification passes. Geometry and its detector remain float64; attention and FFN matrices are float32. The saved arrays are checked for exact equality with the verified arrays. A failed check returns a nonzero exit code.

Verify a saved artifact:

```powershell
python .\compile_weights.py verify .\compiled_output\weights.npz
```

The weight artifact includes the numerical model arrays and the supporting aggregate evidence arrays. It does not include a target-free generation loop.

## Verification performed

The control dataset was the actual Python source of `tools/254d_profiler_objectives_all_v2.py` at the source commit below. Every nonempty physical line became one `CODE_CAUSAL` record. Line endings served as record boundaries. This produced **209 records and 8,985 ground-truth character occurrences**.

Actual input lines included:

```python
def assert_ascii95(text: str, source_id: str = "record") -> None:
def ground_truth_positions(record: dict) -> list[int]:
def visible_positions(objective: str, position: int, n: int) -> tuple[int, ...]:
```

| Check | Result |
| --- | --- |
| Extracted equation functions | 12, AST-equivalent after the documented namespace/precision changes |
| Independent explicit-view enumeration | Exact equality with vectorized near/far/objective/target counts across all six masks and dense/sparse truth selections |
| Compiled embedding rank | 95 |
| Saved detector maximum absolute error | 7.4804 × 10⁻¹⁵ |
| Q/K score maximum absolute error | 2.3837 × 10⁻⁷ |
| Signed FFN matrix maximum absolute error | 9.2371 × 10⁻¹⁴ |
| Embedding → post-attention → post-FFN residual RMS | 0.0344030 → 0.0189073 → 0.0164325 |
| Saved arrays match verified arrays | Exact |
| Deliberately corrupted embedding | Rejected |
| Noninteger, duplicate, unsorted, out-of-range truth positions | Rejected |

The nonzero final residual is retained. The source's float32 embedding/detector combination produced an error of `3.6604e-6` on these same inputs, exceeding its own `1e-8` verification threshold. Preserving both arrays in float64 repairs that numerical inconsistency without changing the embedding equation. The extracted compiler also verifies the arrays that are actually serialized and refuses to publish failed weights.

These results concern the historical equations and their numerical realization. Generalization, conversational output, unknown-character prediction, arbitrary architectures, and full-scale dataset performance were not established by this run.

## Remaining canonical compilation work

The workbook is authoritative. The next implementation must bind its typed fields to actual occurrence evidence before numerical compilation:

1. Preserve each canonical ID, interaction role, participation gate, and declared defer axis.
2. Bind values from source evidence. Resolve derived fields through their declared dependencies. Retain typed relations and distinguish missing evidence from negative evidence.
3. Specify how each participating field enters the selected architecture: coordinate, gate, relation projection, contextual input, or evidence modulation. Keys remain identifiers. A count of 437 registry columns does not prescribe 437 free embedding coordinates.
4. Define the joint target/competitor constraints and an executable inference equation that uses available input context. The unavailable target must stay outside runtime inputs.
5. Compile the shared geometry, freeze it, derive contextual residual requirements, compile attention, freeze it, then compile remaining FFN requirements. Verify those exact frozen weights against the declared requirements and report unresolved constraints.

The record snapshot in this extraction preserves literal input and metadata. Aggregate counting does not implement all arena analyses, occurrence-specific semantic bindings, rank transitions, or cross-analysis reconciliation. The unimplemented pieces are not replaced with guessed cell values or an unrelated solver.

## Provenance

Source repository: `natesoundz/Compiled-intelligence-lab`.

Source branch: `matrix-compiled-254-attention-ffn`.

Pinned commit: `27411dc16e3cb7e286bf3eec9708dae88f62ebc4`.

Extracted from `tools/compile_all_objectives_254.py` with the required objective definitions from `tools/254d_profiler_objectives_all_v2.py`. The workbook SHA-256 is `065cb5424ec0fd2d206de093e13ed2317f29a0d6a1eda837df1a9d46f81506ab`.

Copyright © 2026 Nathan / Nate Soundz. Compiled Intelligence Lab.
