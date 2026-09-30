#!/usr/bin/env python3
"""Conditional expert-path runtime over ASCII95 compiler weights.

No optimizer. One persistent state. Attention is optional; the compiled
per-character FFN correction field is partitioned into experts. Normalize once
at exit, then decode against the frozen token geometry.
"""
from __future__ import annotations
import argparse, json
import numpy as np

ASCII_MIN = 32
V = 95

def softmax(x):
    x = np.asarray(x, np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def final_norm(h, eps=1e-8):
    mu = h.mean()
    return (h - mu) / np.sqrt(np.mean((h - mu) ** 2) + eps)

class ExpertPath:
    def __init__(self, weights):
        z = np.load(weights, allow_pickle=False)
        self.E = np.asarray(z["token_geometry"], np.float64)
        self.mutual = np.asarray(z["mutual_relation"], np.float64)
        self.WV = np.asarray(z["attention_WV"], np.float64)
        self.corr = np.asarray(z["character_ffn_correction"], np.float64)
        self.offsets = np.asarray(z["offsets"], np.int64)
        if self.E.shape[0] != V or self.corr.shape != self.E.shape:
            raise ValueError("expected ASCII95 compiler geometry and correction field")
        self.D = self.E.shape[1]

    def relations(self, text):
        rel = []
        n = len(text)
        for oi, d in enumerate(self.offsets.tolist()):
            if d >= 0:
                continue
            j = n + d
            if 0 <= j < n:
                rel.append((oi, ord(text[j]) - ASCII_MIN))
        return rel

    def field(self, rels):
        if not rels:
            return np.full(V, 1.0 / V), np.empty(0)
        survivors = np.ones(V, bool)
        for oi, c in rels:
            survivors &= self.mutual[oi, c] > 0
        if not survivors.any():
            return np.full(V, 1.0 / V), np.zeros(len(rels))
        idx = np.flatnonzero(survivors)
        logs = [np.log(self.mutual[oi, c, idx]) for oi, c in rels]
        L = np.stack(logs)
        qv = softmax(L.mean(0))
        q = np.zeros(V)
        q[idx] = qv
        relation_scores = np.asarray([np.dot(q[idx], row) for row in L])
        alpha = softmax(relation_scores)
        return q, alpha

    def step(self, text, max_experts=4, attention=True):
        rels = self.relations(text)
        q, alpha = self.field(rels)
        h0 = q @ self.E
        h = h0.copy()
        used_attention = False

        if attention and rels and len(alpha) and np.any(alpha):
            values = np.stack([self.WV[oi, c] for oi, c in rels])
            h = h + alpha @ values
            used_attention = True

        route = np.argsort(-q)[:max_experts]
        path = []
        for y in route:
            if q[y] <= 0:
                continue
            delta = (h0 - h) * self.corr[y]
            h = h + q[y] * delta
            path.append(int(y))

        hn = final_norm(h)
        En = np.stack([final_norm(e) for e in self.E])
        scores = hn @ En.T
        return int(np.argmax(scores)), {
            "attention": used_attention,
            "experts": path,
            "q_top": [(int(i), float(q[i])) for i in route],
            "state_norm": float(np.linalg.norm(h)),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompt", default="def add(x, y): return ")
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--experts", type=int, default=4)
    a = ap.parse_args()
    model = ExpertPath(a.weights)
    text = "".join(c for c in a.prompt if 32 <= ord(c) <= 126)
    traces = []
    for _ in range(a.tokens):
        y, trace = model.step(text, a.experts, True)
        text += chr(y + ASCII_MIN)
        traces.append(trace)
    print(text)
    print(json.dumps({
        "generated": a.tokens,
        "last_trace": traces[-1] if traces else None,
    }, indent=2))

if __name__ == "__main__":
    main()
