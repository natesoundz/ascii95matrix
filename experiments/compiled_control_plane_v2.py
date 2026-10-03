#!/usr/bin/env python3
"""Representation-addressable TopoKV/HAR control plane for the compiled ASCII95 runtime.

This is a versioned descendant. It does not modify the compiled runtime or its
weights. It exposes the already-compiled computational surfaces as explicit,
traceable control points:

TopoKV: visible relation composition / relation-value contribution.
HAR:    representation-addressed activation operations on h0, attention delta,
        post-attention residual, FFN delta, and post-FFN residual.
Decoder: candidate eligibility and score control, kept separate from HAR.

Every HAR operation names canonical registry dimensions. Unknown/ineligible
registry cells are never silently interpreted as zero evidence.
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable
import torch

from tools.target_free_torch import (
    ASCII_MIN, ASCII_MAX, V, OFFSET_TO_INDEX, CompiledASCII95Torch, load_runtime
)

class TruthState(str, Enum):
    PRESENT="present"
    ABSENT="absent"
    UNKNOWN="unknown"
    IMPOSSIBLE="impossible"
    INAPPLICABLE="inapplicable"

class RelationState(str, Enum):
    REQUIRED="required"
    PERMITTED="permitted"
    EXCLUDED="excluded"
    CONDITIONAL="conditional"
    UNRESOLVED="unresolved"

class HAROp(str, Enum):
    MAINTAIN="maintain"
    SUPPRESS="suppress"
    AMPLIFY="amplify"
    INVERT="invert"
    REPLACE="replace"
    REROUTE="reroute"

class HARSite(str, Enum):
    EMBEDDING="embedding"
    ATTENTION_DELTA="attention_delta"
    POST_ATTENTION="post_attention"
    FFN_DELTA="ffn_delta"
    POST_FFN="post_ffn"

@dataclass(frozen=True)
class Representation:
    index:int
    id:str
    interaction:str
    defer_scope:str
    status_by_ascii:tuple[str,...]

class Registry437:
    def __init__(self, path:str|Path):
        raw=json.loads(Path(path).read_text(encoding="utf-8"))
        dims=raw["dimensions"]
        if len(dims)!=437:
            raise ValueError(f"expected 437 canonical representations, got {len(dims)}")
        if [int(x["index"]) for x in dims] != list(range(437)):
            raise ValueError("registry indices must be exactly 0..436")
        self.representations=tuple(
            Representation(
                int(x["index"]), str(x["id"]), str(x.get("interaction","")),
                str(x.get("defer_scope","")), tuple(map(str,x["status_by_ascii"]))
            ) for x in dims
        )
        self.by_id={r.id:r for r in self.representations}
        if any(len(r.status_by_ascii)!=95 for r in self.representations):
            raise ValueError("every representation must have 95 ASCII status cells")

    def indices(self, ids:Iterable[str])->tuple[int,...]:
        out=[]
        for rid in ids:
            if rid not in self.by_id: raise KeyError(f"unknown representation {rid!r}")
            out.append(self.by_id[rid].index)
        return tuple(out)

    def cell_status(self, rid:str, ascii_index:int)->str:
        return self.by_id[rid].status_by_ascii[ascii_index]

@dataclass(frozen=True)
class HARAction:
    site:HARSite
    op:HAROp
    representation_ids:tuple[str,...]
    factor:float=1.0
    destination_ids:tuple[str,...]=()
    replacement:tuple[float,...]=()

@dataclass(frozen=True)
class TopoKVAction:
    """Control one compiled context relation, not an HAR activation."""
    offset:int
    context_char:str
    factor:float=1.0
    enabled:bool=True

@dataclass(frozen=True)
class DecoderAction:
    allow_ascii:tuple[int,...]=()
    deny_ascii:tuple[int,...]=()
    score_bias:tuple[tuple[int,float],...]=()

class CompiledControlPlaneV2:
    def __init__(self, model:CompiledASCII95Torch, registry:Registry437,
                 har:Iterable[HARAction]=(), topokv:Iterable[TopoKVAction]=(),
                 decoder:DecoderAction|None=None):
        if model.d_model != 437:
            raise ValueError(f"control plane requires compiled d_model=437, got {model.d_model}")
        self.model=model
        self.registry=registry
        self.har=tuple(har)
        self.topokv=tuple(topokv)
        self.decoder=decoder or DecoderAction()

    def _har(self, site:HARSite, x:torch.Tensor)->tuple[torch.Tensor,list[dict]]:
        y=x.clone(); trace=[]
        for a in self.har:
            if a.site != site: continue
            idx=self.registry.indices(a.representation_ids)
            before=y[list(idx)].clone()
            if a.op is HAROp.MAINTAIN:
                pass
            elif a.op is HAROp.SUPPRESS:
                y[list(idx)] *= a.factor
            elif a.op is HAROp.AMPLIFY:
                y[list(idx)] *= a.factor
            elif a.op is HAROp.INVERT:
                y[list(idx)] *= -1.0
            elif a.op is HAROp.REPLACE:
                if len(a.replacement)!=len(idx): raise ValueError("replacement length mismatch")
                y[list(idx)]=torch.tensor(a.replacement,dtype=y.dtype,device=y.device)
            elif a.op is HAROp.REROUTE:
                dst=self.registry.indices(a.destination_ids)
                if len(dst)!=len(idx): raise ValueError("reroute source/destination length mismatch")
                moved=y[list(idx)]*a.factor
                y[list(idx)]-=moved; y[list(dst)]+=moved
            trace.append({"site":site.value,"op":a.op.value,"representations":list(a.representation_ids),
                          "indices":list(idx),"before":before.tolist(),"after":y[list(idx)].tolist()})
        return y,trace

    def _topokv_values(self, relations, alpha):
        values=[]; trace=[]
        rules={(OFFSET_TO_INDEX[a.offset],ord(a.context_char)-ASCII_MIN):a for a in self.topokv}
        for oi,c in relations:
            v=self.model.attention_value[oi,c].clone()
            a=rules.get((oi,c))
            if a is not None:
                if not a.enabled: v.zero_()
                else: v*=a.factor
            values.append(v)
            trace.append({"offset_index":oi,"context_ascii":c+ASCII_MIN,
                          "factor":None if a is None else a.factor,
                          "enabled":True if a is None else a.enabled})
        if not values:
            return torch.zeros(self.model.d_model,dtype=self.model.E.dtype,device=self.model.E.device),trace
        return alpha @ torch.stack(values,dim=0),trace

    def _decoder(self, scores:torch.Tensor, survivors:torch.Tensor):
        s=scores.clone(); mask=survivors.clone()
        if self.decoder.allow_ascii:
            allowed=torch.zeros(V,dtype=torch.bool,device=s.device)
            for code in self.decoder.allow_ascii:
                if not ASCII_MIN<=code<=ASCII_MAX: raise ValueError("allow_ascii outside ASCII95")
                allowed[code-ASCII_MIN]=True
            mask &= allowed
        for code in self.decoder.deny_ascii:
            if not ASCII_MIN<=code<=ASCII_MAX: raise ValueError("deny_ascii outside ASCII95")
            mask[code-ASCII_MIN]=False
        s=s.masked_fill(~mask,-torch.inf)
        for code,bias in self.decoder.score_bias:
            if mask[code-ASCII_MIN]: s[code-ASCII_MIN]+=float(bias)
        return s,mask

    @torch.no_grad()
    def forward_relations(self, relations, gate_mask=None):
        q,survivors,alpha,contradiction=self.model.constraint_field(relations,gate_mask)
        if contradiction:
            return {"contradiction":True,"winner_index":None,"q":q,"survivor_mask":survivors,"trace":[]}
        trace=[]
        h0=q@self.model.E
        h0,t=self._har(HARSite.EMBEDDING,h0); trace+=t
        delta_a,tkv=self._topokv_values(relations,alpha)
        delta_a,t=self._har(HARSite.ATTENTION_DELTA,delta_a); trace+=t
        hA=h0+delta_a
        hA,t=self._har(HARSite.POST_ATTENTION,hA); trace+=t
        corr=q@self.model.ffn_corr
        delta_ffn=corr*(h0-hA)
        delta_ffn,t=self._har(HARSite.FFN_DELTA,delta_ffn); trace+=t
        hF=hA+delta_ffn
        hF,t=self._har(HARSite.POST_FFN,hF); trace+=t
        scores=-torch.sum((self.model.E-hF.unsqueeze(0))**2,dim=1)
        scores= scores.masked_fill(~survivors,-torch.inf)
        scores,final_mask=self._decoder(scores,survivors)
        winner=None if not bool(final_mask.any()) else int(torch.argmax(scores).item())
        return {"contradiction":False,"q":q,"survivor_mask":final_mask,"attention_weights":alpha,
                "h0":h0,"attention_delta":delta_a,"hA":hA,"ffn_delta":delta_ffn,"hF":hF,
                "scores":scores,"winner_index":winner,"topokv_trace":tkv,"trace":trace}

    @torch.no_grad()
    def predict_next(self,prefix:str,gate_mask=None):
        if any(not ASCII_MIN<=ord(ch)<=ASCII_MAX for ch in prefix):
            raise ValueError("prefix must contain printable ASCII95 only")
        rel=[]
        n=len(prefix)
        for dist in range(1,8):
            j=n-dist
            if j<0: break
            rel.append((OFFSET_TO_INDEX[-dist],ord(prefix[j])-ASCII_MIN))
        r=self.forward_relations(rel,gate_mask)
        r["winner_ascii"]=None if r["winner_index"] is None else r["winner_index"]+ASCII_MIN
        r["winner"]=None if r["winner_ascii"] is None else chr(r["winner_ascii"])
        return r

def action_from_json(x):
    return HARAction(HARSite(x["site"]),HAROp(x["op"]),tuple(x["representation_ids"]),
                     float(x.get("factor",1.0)),tuple(x.get("destination_ids",[])),
                     tuple(map(float,x.get("replacement",[]))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runtime",required=True); ap.add_argument("--registry",default="data/registry_contract.json")
    ap.add_argument("--prefix",required=True); ap.add_argument("--config")
    a=ap.parse_args(); cfg={} if not a.config else json.loads(Path(a.config).read_text())
    har=[action_from_json(x) for x in cfg.get("har",[])]
    top=[TopoKVAction(int(x["offset"]),str(x["context_char"]),float(x.get("factor",1.0)),bool(x.get("enabled",True)))
         for x in cfg.get("topokv",[])]
    d=cfg.get("decoder",{})
    dec=DecoderAction(tuple(d.get("allow_ascii",[])),tuple(d.get("deny_ascii",[])),
                      tuple((int(k),float(v)) for k,v in d.get("score_bias",{}).items()))
    cp=CompiledControlPlaneV2(load_runtime(a.runtime),Registry437(a.registry),har,top,dec)
    r=cp.predict_next(a.prefix)
    print(json.dumps({"winner":r["winner"],"winner_ascii":r["winner_ascii"],
                      "contradiction":r["contradiction"],"topokv_trace":r.get("topokv_trace",[]),
                      "har_trace":r.get("trace",[])},indent=2))
if __name__=="__main__": main()
