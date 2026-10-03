from __future__ import annotations

import argparse, json, time
from pathlib import Path

from tools.target_free_torch import ASCII_MIN, ground_truth_positions, iter_records, load_runtime
from experiments.har_topokv_compiled_v1 import CompiledHARV1, HARIntervention, Op, Stage


def cases(dataset):
    for record in iter_records(dataset):
        text = record["text"]
        for p in ground_truth_positions(record):
            if p:
                yield text[:p], ord(text[p]) - ASCII_MIN


def measure(predict, dataset):
    n = correct = 0
    t0 = time.perf_counter()
    for prefix, target in cases(dataset):
        out = predict(prefix)
        if out.get("winner_index") is None:
            continue
        n += 1
        correct += int(out["winner_index"] == target)
    sec = time.perf_counter() - t0
    return {"occurrences": n, "top1": correct/max(1,n), "seconds": sec,
            "microseconds_per_occurrence": sec*1e6/max(1,n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--test-evidence", required=True,
                    help="Must be disjoint from the evidence used to compile the runtime.")
    ap.add_argument("--report", default="har_topokv_compiled_v2_report.json")
    a = ap.parse_args()
    model = load_runtime(a.runtime, device="cpu").eval()
    data = Path(a.test_evidence)

    raw = measure(model.predict_next, data)
    observe = CompiledHARV1(model)
    obs = measure(observe.predict_next, data)
    probe = CompiledHARV1(model, (
        HARIntervention(Stage.POST_ATTENTION, Op.SUPPRESS, (0,), 0.5),
    ))
    ctl = measure(probe.predict_next, data)

    report = {
        "runtime": str(Path(a.runtime).resolve()),
        "test_evidence": str(data.resolve()),
        "generalization_contract": "test evidence must be disjoint from runtime compilation evidence",
        "raw_baseline": raw,
        "har_observation": obs,
        "har_control_probe": ctl,
        "observation_top1_delta": obs["top1"]-raw["top1"],
        "control_top1_delta": ctl["top1"]-raw["top1"],
        "observation_latency_overhead_us": obs["microseconds_per_occurrence"]-raw["microseconds_per_occurrence"],
        "control_latency_overhead_us": ctl["microseconds_per_occurrence"]-raw["microseconds_per_occurrence"],
        "claims": {
            "observation_preserves_capability": obs["top1"] == raw["top1"],
            "control_changes_behavior": ctl["top1"] != obs["top1"],
            "capability_improved_by_probe": ctl["top1"] > raw["top1"],
            "generalization_improved_by_probe": ctl["top1"] > raw["top1"],
        },
        "interpretation": "The fixed suppression rule is a falsifiable control probe, not a tuned policy. No improvement is claimed unless the held-out measurement is positive."
    }
    Path(a.report).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
