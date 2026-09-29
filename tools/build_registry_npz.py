#!/usr/bin/env python3
"""Build the NPZ registry interface required by relational_compiler.py.

The canonical JSON remains authoritative. Numeric participation is true only
where a dimension may receive a numeric occurrence measurement. X and K are
excluded: X is ineligible and K is identity metadata, not a weight input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ASCII_MIN = 32
ASCII_MAX = 126
V = 95
D = 437


def build(contract_path: Path, out: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    dims = contract["dimensions"]
    if len(dims) != D:
        raise ValueError(f"expected {D} dimensions")
    ascii_code = np.arange(ASCII_MIN, ASCII_MAX + 1, dtype=np.int16)
    dimension_name = np.asarray([d["id"] for d in dims], dtype=str)
    participation = np.zeros((V, D), dtype=bool)
    for d in dims:
        k = int(d["index"])
        statuses = d["status_by_ascii"]
        if len(statuses) != V:
            raise ValueError(f"{d['id']}: expected 95 status cells")
        participation[:, k] = np.asarray(
            [s not in ("X", "K") for s in statuses],
            dtype=bool,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ascii_code=ascii_code,
        dimension_name=dimension_name,
        participation=participation,
    )
    return {
        "output": str(out),
        "characters": V,
        "dimensions": D,
        "numeric_participation_cells": int(participation.sum()),
        "excluded_cells": int(participation.size - participation.sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--contract",
        type=Path,
        default=Path("data/registry_contract.json"),
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        report = build(args.contract, args.out)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
