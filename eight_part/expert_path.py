#!/usr/bin/env python3
"""Eight-part conditional-path runtime over canonical ASCII95 95x437 geometry.

This runtime refuses legacy 254-D weights. It consumes only a frozen directory
produced by relational_compiler.py. No Q/K/V or FFN weights are fabricated:
until those 437-D downstream operators are compiled, the executable path is
limited to geometry validation and canonical decoder self-correspondence.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relational_compiler import FrozenGeometry, validate_frozen, decode_character

D = 437
V = 95

class CanonicalExpertPath:
    def __init__(self, frozen_directory: str | Path):
        self.frozen = FrozenGeometry(frozen_directory)
        if self.frozen.E.shape != (V, D):
            raise ValueError(
                f"canonical eight-part runtime requires E[95,437], got {self.frozen.E.shape}"
            )

    def self_decode_all(self) -> dict:
        errors = []
        for i in range(V):
            ch, code, _ = decode_character(
                self.frozen.E[i],
                self.frozen,
                correspondence="negative_squared_distance",
            )
            expected = int(self.frozen.ascii_code[i])
            if code != expected:
                errors.append({"index": i, "expected": expected, "got": code, "character": ch})
        return {
            "geometry_shape": list(self.frozen.E.shape),
            "decoded": V,
            "errors": errors,
            "passed": not errors,
        }

    def generate(self, prompt: str, tokens: int) -> str:
        raise RuntimeError(
            "437-D Q/K/V and FFN expert operators are not compiled by relational_compiler.py yet; "
            "generation is blocked rather than substituting legacy 254-D operators."
        )

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--self-decode", action="store_true")
    ap.add_argument("--prompt")
    ap.add_argument("--tokens", type=int, default=1)
    a = ap.parse_args()
    model = CanonicalExpertPath(a.frozen)
    report = {"frozen_validation": validate_frozen(a.frozen)}
    if a.self_decode:
        report["self_decode"] = model.self_decode_all()
    print(json.dumps(report, indent=2))
    if a.prompt is not None:
        model.generate(a.prompt, a.tokens)
    return 0 if report.get("self_decode", {"passed": True})["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
