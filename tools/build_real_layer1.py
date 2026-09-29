#!/usr/bin/env python3
"""One-command strict real layer-one build orchestrator.

The build always creates the annotated arena and coverage report first.
It refuses to compile geometry while any canonical dimension lacks warranted
coverage. When the 437/437 gate passes, it builds registry.npz, freezes the
geometry with relational_compiler.py, and then invokes the target-free PyTorch
compiler if supplied.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path.home() / "Downloads" / "webster_full_lexicon.sqlite"


def run(cmd: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lexicon-db", type=Path, default=DEFAULT_DB)
    p.add_argument("--corpus", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--target-free-script",
        type=Path,
        default=ROOT / "tools" / "target_free_torch.py",
    )
    args = p.parse_args()

    out = args.out.resolve()
    arena = out / "arena"
    evidence = out / "scalar_evidence"
    registry_npz = out / "registry.npz"
    frozen = out / "frozen_geometry"
    runtime = out / "target_free_runtime"

    out.mkdir(parents=True, exist_ok=True)
    corpora = args.corpus or [ROOT]

    assemble = [
        sys.executable,
        str(ROOT / "tools" / "assemble_real_dataset.py"),
        "--lexicon-db",
        str(args.lexicon_db),
        "--out",
        str(arena),
    ]
    for corpus in corpora:
        assemble.extend(["--corpus", str(corpus)])
    run(assemble)

    run(
        [
            sys.executable,
            str(ROOT / "tools" / "project_scalar_evidence.py"),
            "--records",
            str(arena / "records.jsonl"),
            "--registry",
            str(ROOT / "data" / "registry_contract.json"),
            "--out",
            str(evidence),
            "--allow-incomplete",
        ]
    )

    coverage = json.loads(
        (evidence / "binding_coverage.json").read_text(encoding="utf-8")
    )

    checkpoint = {
        "arena_manifest": str(arena / "manifest.json"),
        "coverage_report": str(evidence / "binding_coverage.json"),
        "strict_full_437_ready": coverage["strict_full_437_ready"],
        "zero_coverage_dimension_count": len(coverage["zero_coverage_dimensions"]),
        "partial_character_coverage_dimension_count": len(
            coverage["partial_character_coverage_dimensions"]
        ),
        "geometry_compiled": False,
        "target_free_runtime_compiled": False,
    }

    if not coverage["strict_full_437_ready"]:
        (out / "build_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(checkpoint, indent=2), flush=True)
        print(
            "BLOCKED: full 437-dimensional compilation requires explicit "
            "bindings/projection laws for the dimensions listed in the coverage report.",
            flush=True,
        )
        return 2

    run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_registry_npz.py"),
            "--contract",
            str(ROOT / "data" / "registry_contract.json"),
            "--out",
            str(registry_npz),
        ]
    )

    shards = sorted(evidence.glob("evidence_*.npz"))
    if not shards:
        raise RuntimeError("projection produced no evidence shards")

    compile_cmd = [
        sys.executable,
        str(ROOT / "relational_compiler.py"),
        "compile",
        "--registry",
        str(registry_npz),
        "--evidence",
        *[str(x) for x in shards],
        "--out",
        str(frozen),
        "--phi",
        "abs",
        "--center-weight",
        "occurrence_weight_mass",
    ]
    run(compile_cmd)
    checkpoint["geometry_compiled"] = True
    checkpoint["frozen_geometry"] = str(frozen)

    if not args.target_free_script.exists():
        checkpoint["target_free_blocker"] = (
            f"missing target-free PyTorch compiler: {args.target_free_script}"
        )
        (out / "build_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(checkpoint, indent=2), flush=True)
        return 3

    run(
        [
            sys.executable,
            str(args.target_free_script),
            "compile",
            "--dataset",
            str(arena / "records.jsonl"),
            "--geometry-dir",
            str(frozen),
            "--out",
            str(runtime),
        ]
    )
    checkpoint["target_free_runtime_compiled"] = True
    checkpoint["target_free_runtime"] = str(runtime)

    (out / "build_checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checkpoint, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
