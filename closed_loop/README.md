# Closed-Loop Compiler v1

This directory is the first implementation of the forward architecture:

```text
VERIFIED EVIDENCE
  -> canonical 437-D geometry
  -> compile candidate attention / FFN
  -> target-hidden replay
  -> TopoKV execution
  -> HAR causal attacks
  -> acceptance gates
  -> FREEZE or REJECT
```

It is an orchestration layer over the existing real-data compiler and the
representation-addressable TopoKV/HAR runtime. Existing compiler/runtime files
are not modified.

## Inputs

The runner requires three real artifacts:

1. a frozen canonical `geometry.npz` + `frozen_manifest.json`,
2. a compilation JSONL evidence set,
3. a distinct evaluation JSONL evidence set.

The compilation and evaluation files are hash-checked and may not be identical.

## Causal tests

For each active canonical representation ID, the gate executes HAR attacks at
the compiled attention and FFN contribution sites:

- suppression: factor 0.25 (not zero ablation),
- amplification: factor 2.0,
- inversion: sign flip.

The baseline and attacked models execute the same TopoKV relation composition.
No compiled model tensor is mutated. Logit control is not used as HAR.

The report records accuracy and target-margin effects for every attack. A
nonzero compiled attention/FFN family must demonstrate a measurable causal
effect on the evaluation slice or its causal gate fails.

Embedding geometry is audited separately for actual numerical spread. This v1
does not yet claim that the complete semantic meaning of a canonical dimension
has been causally validated merely because it has numerical spread.

## Leakage gate

Ground truth is available to the existing compiler while it solves candidate
components. The closed-loop evaluation path receives only the context relations
and candidate gate available to the target-free runtime. The target is used
after execution only for scoring.

A candidate can be frozen only after evaluation on a distinct evidence file.

## Decision

`closed_loop_report.json` contains every gate and returns exactly one decision:

- `FREEZE`
- `REJECT`

On FREEZE, the exact runtime artifact is copied into `frozen/` and a
hash-linked freeze manifest is written.

## Run

```powershell
python closed_loop/closed_loop_compiler_v1.py --compile-dataset <compile.jsonl> --eval-dataset <heldout.jsonl> --geometry-dir <frozen_geometry_dir> --out <new_empty_output_dir>
```

By default all active canonical representations are attacked and any
contradiction in held-out replay rejects the candidate.

## Deliberately not claimed yet

v1 does not yet compile new operator families from phonetic/morphological/
grammar/math transition deltas. It validates the current compiled attention and
field-fired FFN machinery under the new closed-loop rule. The next compiler
stage should derive recurrent explicit-state transformations, assign their
conditions/invariants, and submit those operators to the same attack/freeze
protocol.
