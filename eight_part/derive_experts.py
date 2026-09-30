#!/usr/bin/env python3
"""Derive expert groups from the canonical ASCII95 x 437 registry.

Grouping is mechanical: the primary expert key is the namespace prefix of each
canonical dimension id (text before the first '.'). Secondary signatures retain
the registry's interaction, defer scope and ASCII participation/status pattern.
Nothing is assigned by character identity or by a hand-written expert list.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

D=437
V=95

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def derive(registry_path: Path):
    r=json.loads(registry_path.read_text(encoding="utf-8"))
    dims=r["dimensions"]
    if len(dims)!=D or [int(x["index"]) for x in dims]!=list(range(D)):
        raise ValueError("registry must contain canonical indices 0..436")
    groups=defaultdict(list)
    signatures=[]
    for d in dims:
        did=str(d["id"])
        family=did.split(".",1)[0]
        groups[family].append(int(d["index"]))
        statuses=tuple(d["status_by_ascii"])
        if len(statuses)!=V:
            raise ValueError(f"{did}: expected 95 status cells")
        signatures.append({
            "index":int(d["index"]),"id":did,"family":family,
            "interaction":str(d.get("interaction","")),
            "defer_scope":str(d.get("defer_scope","")),
            "eligible_ascii":list(map(int,d.get("eligible_ascii",[]))),
            "status_pattern_sha256":hashlib.sha256("\0".join(statuses).encode()).hexdigest(),
        })
    ordered={k:groups[k] for k in sorted(groups)}
    covered=sorted(i for xs in ordered.values() for i in xs)
    if covered!=list(range(D)):
        raise RuntimeError("expert derivation failed exact 437 coverage")
    mask=np.zeros((len(ordered),D),dtype=np.uint8)
    names=list(ordered)
    for gi,name in enumerate(names):
        mask[gi,ordered[name]]=1
    return {
        "format":"ascii95-registry-derived-experts-v1",
        "registry_sha256":sha256(registry_path),
        "derivation":"primary expert = exact namespace prefix of canonical dimension id",
        "expert_count":len(names),
        "dimension_count":D,
        "exact_coverage":True,
        "overlap_at_primary_level":False,
        "experts":[{"name":n,"count":len(ordered[n]),"indices":ordered[n]} for n in names],
        "dimension_signatures":signatures,
    }, names, mask

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--registry",type=Path,default=Path("data/registry_contract.json"))
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    report,names,mask=derive(a.registry)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"experts.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    np.savez_compressed(a.out/"expert_masks.npz",expert_name=np.asarray(names),mask=mask)
    print(json.dumps({"expert_count":report["expert_count"],"dimension_count":D,"exact_coverage":True,
      "groups":{x["name"]:x["count"] for x in report["experts"]}},indent=2))
if __name__=="__main__": main()
