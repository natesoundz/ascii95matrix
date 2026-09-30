#!/usr/bin/env python3
"""Compile canonical 437-D downstream operators from frozen ASCII95 geometry.

The laws use only quantities already frozen by relational_compiler.py:
E[95,437] and rho[95,437]. No legacy 254-D arrays are consumed.

Q = E
K = E
V = E
FFN expert c writes toward E[c] with a diagonal evidence-specific gate rho[c].

This is a deterministic geometry-derived baseline. It is deliberately separate
from future occurrence-sequence Q/K/V compilation so its provenance is exact.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from relational_compiler import FrozenGeometry

V=95
D=437

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def compile_ops(frozen_dir: Path, out: Path) -> dict:
    f=FrozenGeometry(frozen_dir)
    if f.E.shape != (V,D):
        raise ValueError(f"expected E[95,437], got {f.E.shape}")
    Q=f.E.copy()
    K=f.E.copy()
    Vv=f.E.copy()
    expert_gate=f.rho.copy()
    # Decoder uses the same frozen geometry as relational_compiler correspondence.
    decoder=f.E.copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,Q=Q,K=K,V=Vv,expert_gate=expert_gate,decoder=decoder)
    report={
        "format":"ascii95-canonical-437-downstream-v1",
        "geometry_shape":list(f.E.shape),
        "Q_shape":list(Q.shape),"K_shape":list(K.shape),"V_shape":list(Vv.shape),
        "expert_gate_shape":list(expert_gate.shape),
        "decoder_shape":list(decoder.shape),
        "all_finite":bool(all(np.isfinite(x).all() for x in (Q,K,Vv,expert_gate,decoder))),
        "rho_sum_max_error":float(np.max(np.abs(expert_gate.sum(axis=0)-1.0))),
        "legacy_254_consumed":False,
        "laws":{
            "Q":"E","K":"E","V":"E",
            "expert_c":"h <- h + strength * rho[c] * (E[c]-h)",
            "decoder":"negative squared distance to E after final normalization"
        }
    }
    report["passed"]=report["all_finite"] and report["rho_sum_max_error"] < 1e-10
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frozen",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--report",required=True)
    a=p.parse_args()
    report=compile_ops(Path(a.frozen),Path(a.out))
    report["operators_sha256"]=sha(Path(a.out))
    Path(a.report).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["passed"] else 1

if __name__=="__main__":
    raise SystemExit(main())
