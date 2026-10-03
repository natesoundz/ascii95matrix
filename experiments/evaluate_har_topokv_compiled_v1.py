from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tools.target_free_torch import ASCII_MIN, ground_truth_positions, iter_records, load_runtime
from experiments.har_topokv_compiled_v1 import CompiledHARV1, HARIntervention, Op, Stage


def evaluate(model, dataset: Path, interventions=()):
    controlled = CompiledHARV1(model, interventions)
    n = correct = 0
    elapsed = 0.0
    for record in iter_records(dataset):
        text = record["text"]
        for p in ground_truth_positions(record):
            prefix = text[:p]
            if not prefix:
                continue
            t0 = time.perf_counter()
            out = controlled.predict_next(prefix)
            elapsed += time.perf_counter() - t0
            if out.get("winner_index") is None:
                continue
            n += 1
            correct += int(out["winner_index"] == ord(text[p]) - ASCII_MIN)
    return {
        "occurrences": n,
        "top1": correct / max(1, n),
        "seconds": elapsed,
        "microseconds_per_occurrence": elapsed * 1e6 / max(1, n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--test-evidence", required=True,
                    help="Evidence not used to compile --runtime for a genuine generalization test.")
    ap.add_argument("--report", default="har_topokv_compiled_v1_report.json")
    args = ap.parse_args()

    model = load_runtime(args.runtime, device="cpu").eval()
    dataset = Path(args.test_evidence)

    baseline = evaluate(model, dataset)
    observation = evaluate(model, dataset, ())

    # Control demonstration: deterministic nonzero suppression of one explicit
    # representation coordinate. This is not asserted to improve accuracy.
    controlled = evaluate(model, dataset, (
        HARIntervention(Stage.POST_ATTENTION, Op.SUPPRESS, (0,), 0.5),
    ))

    report = {
        "runtime": str(Path(args.runtime).resolve()),
        "test_evidence": str(dataset.resolve()),
        "generalization_contract": "test evidence must be disjoint from runtime compilation evidence",
        "baseline": baseline,
        "har_observation": observation,
        "har_control_demo": controlled,
        "observation_accuracy_delta": observation["top1"] - baseline["top1"],
        "control_accuracy_delta": controlled["top1"] - baseline["top1"],
        "observation_latency_overhead_us": observation["microseconds_per_occurrence"] - baseline["microseconds_per_occurrence"],
        "control_latency_overhead_us": controlled["microseconds_per_occurrence"] - baseline["microseconds_per_occurrence"],
        "claims": {
            "capability_improved": controlled["top1"] > baseline["top1"],
            "generalization_improved": controlled["top1"] > baseline["top1"],
            "control_surface_active": controlled["top1"] != observation["top1"],
            "observation_preserves_capability": observation["top1"] == baseline["top1"],
        },
        "note": "A fixed suppression rule is a control probe, not an optimized policy. Improvement is reported only if measured.",
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
