#!/usr/bin/env python3
"""Create an explicit per-dimension binding manifest from registry_contract.json.

This does not guess deferred measurements. It binds only facts already declared
by the canonical workbook (1/0/X/K) and marks E/R/M/U/V/S cases as requiring an
explicit occurrence-level resolver before a strict 437-D compile can pass.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

UNBOUND = {"E", "R", "M", "U", "V", "S"}


def build(contract: dict) -> dict:
    dimensions = []
    unresolved_status_counts = Counter()
    dimensions_requiring_resolver = 0

    for d in contract["dimensions"]:
        statuses = set(d["status_by_ascii"])
        unresolved = sorted(statuses & UNBOUND)
        for s in unresolved:
            unresolved_status_counts[s] += d["status_counts"].get(s, 0)
        if unresolved:
            dimensions_requiring_resolver += 1

        dimensions.append(
            {
                "index": d["index"],
                "id": d["id"],
                "name": d["name"],
                "interaction": d["interaction"],
                "eligibility": d["eligibility"],
                "defer_scope": d["defer_scope"],
                "status_by_ascii": d["status_by_ascii"],
                "status_dispatch": {
                    "1": {"resolver": "CONSTANT", "value": 1.0},
                    "0": {"resolver": "CONSTANT", "value": 0.0},
                    "X": {"resolver": "UNAVAILABLE", "value": None},
                    "K": {"resolver": "IDENTITY_KEY_ONLY", "value": None},
                    "E": {"resolver": None},
                    "R": {"resolver": None},
                    "M": {"resolver": None},
                    "U": {"resolver": None},
                    "V": {"resolver": None},
                    "S": {"resolver": None},
                },
                "required_occurrence_resolvers": unresolved,
                "measurement_units": None,
                "transform": None,
                "source_requirements": [],
                "notes": "",
            }
        )

    return {
        "format": "ascii95-binding-manifest-v1",
        "registry_sha256": contract["source_sha256"],
        "dimension_count": contract["dimension_count"],
        "strict_compile_ready": dimensions_requiring_resolver == 0,
        "dimensions_requiring_occurrence_resolver": dimensions_requiring_resolver,
        "unresolved_status_cell_counts": dict(sorted(unresolved_status_counts.items())),
        "rules": {
            "no_guessing": True,
            "missing_is_not_zero": True,
            "identity_key_is_not_numeric_measurement": True,
            "deferred_relation_measurement_modulator_value_structural_require_explicit_binding": True,
            "all_numeric_measurements_require_provenance": True,
        },
        "dimensions": dimensions,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, default=Path("data/registry_contract.json"))
    p.add_argument("--out", type=Path, default=Path("data/binding_manifest.json"))
    args = p.parse_args()
    contract = json.loads(args.registry.read_text(encoding="utf-8"))
    manifest = build(contract)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "dimensions": manifest["dimension_count"],
        "strict_compile_ready": manifest["strict_compile_ready"],
        "dimensions_requiring_occurrence_resolver": manifest["dimensions_requiring_occurrence_resolver"],
        "unresolved_status_cell_counts": manifest["unresolved_status_cell_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
