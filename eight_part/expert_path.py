#!/usr/bin/env python3
"""Conditional expert-path runtime using only canonical ASCII95 95x437 arrays."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from relational_compiler import FrozenGeometry

V=95; D=437; ASCII_MIN=32

def softmax(x):
    x=np.asarray(x,np.float64); x=x-np.max(x); e=np.exp(x); return e/e.sum()

def norm(x,eps=1e-12):
    x=np.asarray(x,np.float64); m=x.mean(); return (x-m)/np.sqrt(np.mean((x-m)**2)+eps)

class ExpertPath:
    def __init__(self,frozen,operators):
        self.f=FrozenGeometry(frozen)
        z=np.load(operators,allow_pickle=False)
        self.Q=np.asarray(z["Q"],np.float64); self.K=np.asarray(z["K"],np.float64)
        self.Vv=np.asarray(z["V"],np.float64); self.gate=np.asarray(z["expert_gate"],np.float64)
        self.decoder=np.asarray(z["decoder"],np.float64)
        for name,a,shape in (
            ("E",self.f.E,(V,D)),("Q",self.Q,(V,D)),("K",self.K,(V,D)),
            ("V",self.Vv,(V,D)),("expert_gate",self.gate,(V,D)),("decoder",self.decoder,(V,D))):
            if a.shape!=shape: raise ValueError(f"{name} expected {shape}, got {a.shape}")

    def step(self,text,max_experts=4):
        ids=np.asarray([ord(c)-ASCII_MIN for c in text if 32<=ord(c)<=126],dtype=int)
        if not len(ids): ids=np.asarray([0],dtype=int)
        q=self.Q[ids[-1]]
        K=self.K[ids]
        scores=(K@q)/np.sqrt(D)
        a=softmax(scores)
        h=a@self.Vv[ids]
        used_attention=len(ids)>1
        # Route by correspondence of the attended state to canonical expert centers.
        d2=np.sum((self.f.E-h[None,:])**2,axis=1)
        route=np.argsort(d2)[:max_experts]
        strengths=softmax(-d2[route]/max(1.0,D))
        path=[]
        for c,s in zip(route,strengths):
            h=h+float(s)*self.gate[c]*(self.f.E[c]-h)
            path.append({"expert":int(c),"character":chr(int(c)+ASCII_MIN),"strength":float(s)})
        hn=norm(h)
        En=np.stack([norm(e) for e in self.decoder])
        ds=np.sum((En-hn[None,:])**2,axis=1)
        y=int(np.argmin(ds))
        return y,{"attention":used_attention,"context":int(len(ids)),"experts":path,"state_norm":float(np.linalg.norm(h))}

    def generate(self,prompt,tokens,max_experts):
        text="".join(c for c in prompt if 32<=ord(c)<=126)
        traces=[]
        for _ in range(tokens):
            y,t=self.step(text,max_experts); text+=chr(y+ASCII_MIN); traces.append(t)
        return text,traces

def main():
    p=argparse.ArgumentParser(); p.add_argument("--frozen",required=True); p.add_argument("--operators",required=True)
    p.add_argument("--prompt",default="def add(x, y): return "); p.add_argument("--tokens",type=int,default=80)
    p.add_argument("--experts",type=int,default=4); a=p.parse_args()
    m=ExpertPath(a.frozen,a.operators); text,tr=m.generate(a.prompt,a.tokens,a.experts)
    print(text); print(json.dumps({"generated":a.tokens,"last_trace":tr[-1] if tr else None},indent=2))
if __name__=="__main__": main()
