#!/usr/bin/env python3
"""Closed-loop compiler gate for the canonical ASCII95 x 437 system.

This module does not replace the existing compilers. It orchestrates them:

real evidence -> frozen E -> compile Q/K/V-equivalent relation field + FFN
-> target-hidden replay -> TopoKV execution -> HAR causal attacks
-> accept/reject -> freeze manifest.

Existing source artifacts are never modified. Ground truth may be used while
compiling a candidate, but evaluation and intervention execution receive only
runtime-visible context. A candidate is frozen only when all declared gates pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tools.target_free_torch import (
    ASCII_MIN, V, compile_runtime, gate_for_position, ground_truth_positions,
    iter_records, load_runtime, visible_relations,
)
from experiments.topokv_har_explicit_control_v2 import (
    ExplicitTopoKVHAR, HARController, HAROp, RepresentationControl,
    RepresentationIndex, Site,
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class EvalOccurrence:
    record_id: str
    position: int
    target: int
    relations: tuple[tuple[int, int], ...]
    gate: torch.Tensor


def evaluation_occurrences(dataset: str | Path) -> list[EvalOccurrence]:
    out = []
    for record in iter_records(dataset):
        for p in ground_truth_positions(record):
            out.append(EvalOccurrence(
                record_id=f"{record['dataset']}/{record['id']}",
                position=p,
                target=ord(record["text"][p]) - ASCII_MIN,
                relations=tuple(visible_relations(record, p)),
                gate=torch.from_numpy(gate_for_position(record, p)),
            ))
    if not out:
        raise ValueError("evaluation dataset contains no ground-truth occurrences")
    return out


def score_runtime(runtime, names: list[str], occurrences: list[EvalOccurrence],
                  controls: tuple[RepresentationControl, ...] = ()) -> dict:
    reps = RepresentationIndex(names)
    har = HARController(reps, controls)
    model = ExplicitTopoKVHAR(runtime, names, har)
    correct = 0
    valid = 0
    contradictions = 0
    margin_sum = 0.0
    for occ in occurrences:
        r = model.forward_relations(list(occ.relations), occ.gate)
        if r["contradiction"] or r["winner_index"] is None:
            contradictions += 1
            continue
        valid += 1
        correct += int(r["winner_index"] == occ.target)
        scores = r["scores"]
        target_score = float(scores[occ.target].item())
        other = scores.clone()
        other[occ.target] = -torch.inf
        best_other = float(torch.max(other).item())
        margin_sum += target_score - best_other
    return {
        "occurrences": len(occurrences),
        "valid": valid,
        "contradictions": contradictions,
        "correct": correct,
        "accuracy": correct / max(1, len(occurrences)),
        "valid_accuracy": correct / max(1, valid),
        "mean_target_margin": margin_sum / max(1, valid),
    }


def active_representation_ids(runtime, names: list[str], epsilon: float = 0.0) -> list[str]:
    # A representation is attackable when it participates numerically in at
    # least one compiled runtime component. This does not invent missing evidence.
    e = runtime.E.detach().cpu().numpy()
    a = runtime.attention_value.detach().cpu().numpy()
    f = runtime.ffn_corr.detach().cpu().numpy()
    active = (
        np.any(np.abs(e) > epsilon, axis=0)
        | np.any(np.abs(a) > epsilon, axis=(0, 1))
        | np.any(np.abs(f) > epsilon, axis=0)
    )
    return [names[i] for i in np.flatnonzero(active)]


def causal_attack(runtime, names: list[str], occurrences: list[EvalOccurrence],
                  max_representations: int | None = None) -> dict:
    baseline = score_runtime(runtime, names, occurrences)
    active = active_representation_ids(runtime, names)
    if max_representations is not None:
        active = active[:max_representations]

    attacks = []
    sites = (Site.ATTENTION, Site.FFN)
    specs = (
        (HAROp.SUPPRESS, 0.25),
        (HAROp.AMPLIFY, 2.0),
        (HAROp.INVERT, 1.0),
    )
    for rid in active:
        for site in sites:
            for op, factor in specs:
                control = RepresentationControl(
                    site=site, op=op, representation_ids=(rid,), factor=factor
                )
                result = score_runtime(runtime, names, occurrences, (control,))
                attacks.append({
                    "representation_id": rid,
                    "site": site.value,
                    "operation": op.value,
                    "factor": factor,
                    "accuracy_delta": result["accuracy"] - baseline["accuracy"],
                    "margin_delta": result["mean_target_margin"] - baseline["mean_target_margin"],
                    "result": result,
                })

    # Causal evidence is not defined as "damage only": amplification may improve
    # and inversion may expose directional dependence. Zero-effect dimensions are
    # reported rather than silently declared useful.
    by_rep = {}
    for rid in active:
        rows = [x for x in attacks if x["representation_id"] == rid]
        max_abs_margin = max((abs(x["margin_delta"]) for x in rows), default=0.0)
        max_abs_accuracy = max((abs(x["accuracy_delta"]) for x in rows), default=0.0)
        by_rep[rid] = {
            "max_abs_margin_effect": max_abs_margin,
            "max_abs_accuracy_effect": max_abs_accuracy,
            "demonstrated_runtime_effect": bool(max_abs_margin > 1e-12 or max_abs_accuracy > 0.0),
        }
    return {"baseline": baseline, "active_representations": active,
            "representation_summary": by_rep, "attacks": attacks}


def embedding_geometry_attacks(runtime, names: list[str]) -> dict:
    # Geometry-level attack is deliberately independent of runtime HAR sites.
    # It asks whether an active canonical coordinate has measurable discriminative
    # structure in frozen E. It does not alter or re-fit E.
    E = runtime.E.detach().cpu().numpy()
    rows = []
    for k, rid in enumerate(names):
        col = E[:, k]
        spread = float(np.std(col))
        span = float(np.max(col) - np.min(col))
        distinct = int(np.unique(col).size)
        rows.append({
            "representation_id": rid,
            "std": spread,
            "span": span,
            "distinct_values": distinct,
            "geometrically_active": bool(spread > 0.0),
        })
    return {
        "active_count": sum(int(x["geometrically_active"]) for x in rows),
        "representations": rows,
    }


def run(args) -> dict:
    compile_dataset = Path(args.compile_dataset).resolve()
    eval_dataset = Path(args.eval_dataset).resolve()
    geometry_dir = Path(args.geometry_dir).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)

    if sha256_file(compile_dataset) == sha256_file(eval_dataset):
        raise ValueError("compile and evaluation evidence must be distinct files")

    candidate_dir = out / "candidate"
    compile_report = compile_runtime(compile_dataset, geometry_dir, candidate_dir)
    runtime = load_runtime(candidate_dir / "runtime.pt", args.device)

    # Dimension names come from the same frozen geometry embedded in the runtime
    # lineage; never fabricate REP.xxx aliases for an acceptance run.
    with np.load(geometry_dir / "geometry.npz", allow_pickle=False) as z:
        names = np.asarray(z["dimension_name"]).astype(str).tolist()
    if len(names) != runtime.d_model or len(set(names)) != len(names):
        raise ValueError("canonical representation IDs do not match runtime width")

    occurrences = evaluation_occurrences(eval_dataset)
    baseline = score_runtime(runtime, names, occurrences)
    geometry_attack = embedding_geometry_attacks(runtime, names)
    causal = causal_attack(
        runtime, names, occurrences,
        None if args.max_attack_representations <= 0 else args.max_attack_representations,
    )

    # Acceptance is intentionally conservative. We do NOT require every canonical
    # dimension to affect this evaluation slice; inactive/unresolved dimensions
    # remain visible. We require the candidate to execute target-free, avoid
    # contradictions above threshold, and demonstrate at least one causal effect
    # in each compiled downstream family that claims a nonzero contribution.
    att_nonzero = bool(torch.any(runtime.attention_value != 0).item())
    ffn_nonzero = bool(torch.any(runtime.ffn_corr != 0).item())
    att_effect = any(
        x["site"] == Site.ATTENTION.value
        and (abs(x["margin_delta"]) > 1e-12 or abs(x["accuracy_delta"]) > 0.0)
        for x in causal["attacks"]
    )
    ffn_effect = any(
        x["site"] == Site.FFN.value
        and (abs(x["margin_delta"]) > 1e-12 or abs(x["accuracy_delta"]) > 0.0)
        for x in causal["attacks"]
    )
    contradiction_fraction = baseline["contradictions"] / max(1, baseline["occurrences"])

    gates = {
        "compile_candidate_passed": compile_report.get("status") == "PASS",
        "runtime_target_input_false": compile_report.get("runtime_target_input") is False,
        "distinct_compile_and_eval_evidence": True,
        "canonical_437_ids": len(names) == 437,
        "embedding_has_active_geometry": geometry_attack["active_count"] > 0,
        "attention_claim_has_causal_effect": (not att_nonzero) or att_effect,
        "ffn_claim_has_causal_effect": (not ffn_nonzero) or ffn_effect,
        "contradiction_fraction_within_limit": contradiction_fraction <= args.max_contradiction_fraction,
        "minimum_eval_accuracy_met": baseline["accuracy"] >= args.min_eval_accuracy,
    }
    accepted = all(gates.values())

    report = {
        "format": "ascii95-closed-loop-compiler-v1",
        "decision": "FREEZE" if accepted else "REJECT",
        "engineering_hypothesis": (
            "Compile explicit evidence into executable components, then retain only "
            "components that survive target-hidden replay and causal intervention tests."
        ),
        "compile_dataset_sha256": sha256_file(compile_dataset),
        "eval_dataset_sha256": sha256_file(eval_dataset),
        "geometry_sha256": compile_report.get("geometry_sha256"),
        "runtime_sha256": compile_report.get("runtime_sha256"),
        "runtime_target_input": False,
        "baseline_eval": baseline,
        "embedding_attack": geometry_attack,
        "causal_attack": causal,
        "gates": gates,
        "thresholds": {
            "min_eval_accuracy": args.min_eval_accuracy,
            "max_contradiction_fraction": args.max_contradiction_fraction,
        },
        "unresolved_policy": "visible; never fabricated or silently zero-filled as negative evidence",
    }
    report_path = out / "closed_loop_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if accepted:
        freeze_dir = out / "frozen"
        freeze_dir.mkdir()
        shutil.copy2(candidate_dir / "runtime.pt", freeze_dir / "runtime.pt")
        manifest = {
            "format": "ascii95-closed-loop-freeze-v1",
            "runtime_sha256": sha256_file(freeze_dir / "runtime.pt"),
            "closed_loop_report_sha256": sha256_file(report_path),
            "compile_dataset_sha256": report["compile_dataset_sha256"],
            "eval_dataset_sha256": report["eval_dataset_sha256"],
            "runtime_target_input": False,
        }
        (freeze_dir / "freeze_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compile-dataset", required=True)
    p.add_argument("--eval-dataset", required=True)
    p.add_argument("--geometry-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-eval-accuracy", type=float, default=0.0)
    p.add_argument("--max-contradiction-fraction", type=float, default=0.0)
    p.add_argument("--max-attack-representations", type=int, default=0,
                   help="0 attacks every active canonical representation")
    args = p.parse_args()
    try:
        report = run(args)
        print(json.dumps({
            "decision": report["decision"],
            "baseline_eval": report["baseline_eval"],
            "gates": report["gates"],
        }, indent=2))
        return 0 if report["decision"] == "FREEZE" else 2
    except Exception as exc:
        print(f"CLOSED LOOP FAILED | {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
