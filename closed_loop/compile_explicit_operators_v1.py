#!/usr/bin/env python3
"""Compile recurrent explicit-state transition operators from real lexical evidence.

This stage never invents phonetic feature vectors. It consumes:
  * lexical_snapshot.jsonl produced by tools/assemble_real_dataset.py,
  * a canonical frozen geometry.npz with 437 named coordinates,
  * binding_coverage.json describing which coordinates have real measurements.

Pronunciation rows are retained as symbolic states unless the canonical binding
report demonstrates occurrence measurements for the corresponding realization
coordinates. Unsupported phonetic/articulatory coordinates remain unresolved.

Operators are exact recurrent deltas between resolved explicit states. Candidate
operators are grouped only when their numerical delta and applicability signature
are identical within the declared tolerance. No k-means, learned clustering,
random initialization, optimizer, or silent averaging is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

D = 437


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def load_binding_coverage(path: Path) -> tuple[list[dict[str,Any]],dict[str,dict[str,Any]]]:
    raw=json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("dimensions", raw) if isinstance(raw,dict) else raw
    if not isinstance(rows,list):
        raise ValueError("binding coverage must contain a dimension list")
    by_id={}
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get("id"),str):
            continue
        by_id[row["id"]]=row
    return rows,by_id


def resolved_realization_dimensions(names:list[str], coverage:dict[str,dict[str,Any]]) -> np.ndarray:
    mask=np.zeros(len(names),dtype=bool)
    for i,rid in enumerate(names):
        row=coverage.get(rid,{})
        n=int(row.get("valid_occurrence_measurements") or 0)
        # Realization dimensions must be backed by actual occurrence measurements.
        if n>0 and (rid.startswith("phonetic.") or rid.startswith("pronunciation.")):
            mask[i]=True
    return mask


def pronunciation_signature(row:dict[str,Any]) -> tuple[tuple[str,str],...]:
    # Preserve the DB row itself as symbolic state. Database primary keys and
    # source IDs identify provenance, not pronunciation content.
    ignored={"pronunciation_id","lexeme_id","source_id","created_at","updated_at"}
    return tuple(sorted(
        (str(k),json.dumps(v,sort_keys=True,ensure_ascii=False,default=str))
        for k,v in row.items() if k not in ignored and v is not None
    ))


def iter_lexical(path:Path):
    with path.open(encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            obj=json.loads(line)
            if not isinstance(obj,dict):
                raise ValueError(f"line {line_no}: expected object")
            yield obj


def extract_symbolic_transitions(path:Path) -> tuple[list[dict[str,Any]],Counter]:
    transitions=[]
    stats=Counter()
    for snap in iter_lexical(path):
        lex=snap.get("lexeme") or {}
        lemma=str(lex.get("lemma") or snap.get("normalized") or "")
        facts=snap.get("facts") or {}
        rows=facts.get("pronunciation") or []
        if not isinstance(rows,list):
            continue
        sigs=[]
        for row in rows:
            if not isinstance(row,dict): continue
            sig=pronunciation_signature(row)
            if sig: sigs.append((sig,row))
        # The database may contain variants but does not, by itself, assert a
        # temporal order between arbitrary variants. We therefore record
        # variant relations, not fabricate P0->P1 chronology.
        for i in range(len(sigs)):
            for j in range(i+1,len(sigs)):
                a,ra=sigs[i]; b,rb=sigs[j]
                if a==b: continue
                transitions.append({
                    "kind":"PRONUNCIATION_VARIANT_RELATION",
                    "lemma":lemma,
                    "state_a":dict(a),
                    "state_b":dict(b),
                    "source_a":ra.get("source_id"),
                    "source_b":rb.get("source_id"),
                    "ordered":False,
                })
                stats["pronunciation_variant_relations"]+=1
        stats["lexemes_with_pronunciation"] += int(bool(sigs))
        stats["pronunciation_rows"] += len(sigs)
    return transitions,stats


def geometry_state_map(E:np.ndarray,names:list[str],resolved_mask:np.ndarray) -> dict[int,np.ndarray]:
    # Character geometry can only supply coordinates that are actually resolved.
    # It is not used as a substitute for a pronunciation realization.
    return {code:E[code-32,resolved_mask].copy() for code in range(32,127)}


def exact_operator_groups(transitions:list[dict[str,Any]], tolerance:float) -> list[dict[str,Any]]:
    # Numeric operators are emitted only for transitions carrying explicit
    # resolved vectors. Symbolic pronunciation variant relations remain evidence.
    groups=defaultdict(list)
    for t in transitions:
        if "vector_a" not in t or "vector_b" not in t:
            continue
        a=np.asarray(t["vector_a"],np.float64); b=np.asarray(t["vector_b"],np.float64)
        if a.shape!=b.shape: raise ValueError("transition vector shape mismatch")
        delta=b-a
        if tolerance<=0:
            key=delta.tobytes()
        else:
            key=np.rint(delta/tolerance).astype(np.int64).tobytes()
        groups[key].append((t,delta))
    out=[]
    for rows in groups.values():
        exemplar=rows[0][1]
        out.append({
            "support":len(rows),
            "delta":exemplar.tolist(),
            "transition_kinds":sorted({x[0]["kind"] for x in rows}),
            "examples":[{"lemma":x[0].get("lemma"),"kind":x[0]["kind"]} for x in rows[:20]],
        })
    out.sort(key=lambda x:(-x["support"],x["transition_kinds"]))
    return out


def compile_operators(lexical:Path, geometry:Path, binding:Path, out:Path, tolerance:float) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    out.mkdir(parents=True,exist_ok=True)
    with np.load(geometry,allow_pickle=False) as z:
        E=np.asarray(z["E"],np.float64)
        names=np.asarray(z["dimension_name"]).astype(str).tolist()
    if E.shape!=(95,D) or len(names)!=D or len(set(names))!=D:
        raise ValueError("expected canonical 95 x 437 frozen geometry")

    _,coverage=load_binding_coverage(binding)
    resolved=resolved_realization_dimensions(names,coverage)
    symbolic,stats=extract_symbolic_transitions(lexical)

    evidence_path=out/"symbolic_transitions.jsonl"
    with evidence_path.open("w",encoding="utf-8") as f:
        for t in symbolic:
            f.write(json.dumps(t,ensure_ascii=False,sort_keys=True,default=str)+"\n")

    operators=exact_operator_groups(symbolic,tolerance)
    operator_path=out/"operators.json"
    operator_path.write_text(json.dumps(operators,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

    resolved_ids=[names[i] for i in np.flatnonzero(resolved)]
    deferred_ids=[
        rid for rid in names
        if (rid.startswith("phonetic.") or rid.startswith("pronunciation."))
        and rid not in resolved_ids
    ]
    status = "READY_FOR_CAUSAL_OPERATOR_TEST" if operators else "BLOCKED_UNRESOLVED_REALIZATION"
    report={
        "format":"ascii95-explicit-transition-operator-compiler-v1",
        "status":status,
        "lexical_snapshot_sha256":sha256_file(lexical),
        "geometry_sha256":sha256_file(geometry),
        "binding_coverage_sha256":sha256_file(binding),
        "dimensions":D,
        "resolved_realization_dimensions":len(resolved_ids),
        "resolved_realization_ids":resolved_ids,
        "deferred_realization_dimensions":len(deferred_ids),
        "symbolic_transition_evidence":len(symbolic),
        "compiled_numeric_operator_families":len(operators),
        "stats":dict(stats),
        "invariants":{
            "pronunciation_variants_treated_as_temporal_sequence":False,
            "missing_phonetic_measurements_fabricated":False,
            "character_geometry_used_as_pronunciation_substitute":False,
            "operator_clustering_learned":False,
            "exact_recurrent_deltas_only":True,
        },
        "next_requirement":(
            "Bind real pronunciation occurrences to canonical phonetic/pronunciation "
            "coordinates, then emit explicit resolved state pairs before numeric operators "
            "can enter HAR causal testing."
            if not operators else
            "Submit numeric operator families to closed-loop HAR/TopoKV causal testing."
        ),
    }
    (out/"operator_compile_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lexical-snapshot",type=Path,required=True)
    p.add_argument("--geometry",type=Path,required=True)
    p.add_argument("--binding-coverage",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--tolerance",type=float,default=0.0)
    a=p.parse_args()
    try:
        r=compile_operators(a.lexical_snapshot,a.geometry,a.binding_coverage,a.out,a.tolerance)
        print(json.dumps(r,indent=2))
        return 0 if r["status"]=="READY_FOR_CAUSAL_OPERATOR_TEST" else 2
    except Exception as e:
        print(f"OPERATOR COMPILE FAILED | {e}")
        return 1

if __name__=="__main__":
    raise SystemExit(main())
