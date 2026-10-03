#!/usr/bin/env python3
"""Registry-aware CLI for explicit TopoKV/HAR control.

Loads representation ids directly from data/registry_dimensions.tsv and an
existing compiled runtime.pt. No registry rebuilding and no weight mutation.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from tools.target_free_torch import load_runtime, OFFSET_TO_INDEX, ASCII_MIN
from experiments.topokv_har_explicit_control_v2 import (
    ExplicitTopoKVHAR,HARController,HAROp,RepresentationControl,RepresentationIndex,Site
)

def dimension_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f,delimiter="\t"))
    rows.sort(key=lambda r:int(r["index"]))
    ids=[r["id"] for r in rows]
    if [int(r["index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("registry indices are not contiguous")
    return ids

def relations_from_prefix(prefix: str):
    out=[]; n=len(prefix)
    for dist in range(1,8):
        j=n-dist
        if j<0: break
        out.append((OFFSET_TO_INDEX[-dist],ord(prefix[j])-ASCII_MIN))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--runtime",required=True)
    p.add_argument("--registry",default="data/registry_dimensions.tsv")
    p.add_argument("--prefix",required=True)
    p.add_argument("--site",choices=[x.value for x in Site])
    p.add_argument("--op",choices=[x.value for x in HAROp])
    p.add_argument("--representation",action="append",default=[])
    p.add_argument("--destination",action="append",default=[])
    p.add_argument("--factor",type=float,default=1.0)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()

    model=load_runtime(a.runtime,a.device)
    ids=dimension_ids(Path(a.registry))
    if len(ids)!=model.d_model:
        raise ValueError(f"registry has {len(ids)} representations but runtime width is {model.d_model}")
    reps=RepresentationIndex(ids)
    rules=[]
    if a.site or a.op or a.representation:
        if not (a.site and a.op and a.representation):
            raise ValueError("site, op and representation must be supplied together")
        rules=[RepresentationControl(
            Site(a.site),HAROp(a.op),tuple(a.representation),a.factor,tuple(a.destination)
        )]
    ctl=ExplicitTopoKVHAR(model,ids,HARController(reps,rules))
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=ctl.forward_relations(relations_from_prefix(a.prefix))
    ctl.assert_weights_unchanged(before)
    wi=result["winner_index"]
    report={
        "prefix":a.prefix,
        "contradiction":result["contradiction"],
        "winner":None if wi is None else chr(wi+ASCII_MIN),
        "winner_ascii":None if wi is None else wi+ASCII_MIN,
        "representation_count":len(ids),
        "control":{
            "site":a.site,"op":a.op,"representations":a.representation,
            "destinations":a.destination,"factor":a.factor,
        },
        "trace":[{
            "site":t["site"],"op":t["op"],"representations":t["representations"],
            "before":t["before"].tolist(),"after":t["after"].tolist()
        } for t in result.get("trace",[])],
        "weights_unchanged":True,
    }
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
