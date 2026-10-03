#!/usr/bin/env python3
"""Representation-addressable TopoKV/HAR control for the compiled ASCII95 runtime.

New integration surface only. Existing compiler/runtime files are not modified.

Separation preserved:
  TopoKV: composes visible compiled relation values.
  HAR: redirects/scales/inverts/suppresses representation-attributed contributions.
  LogitControl: constrains output candidates separately.
  Trace: records attribution before/after each control site.

No model weight or compiled table is mutated.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable
import torch

from tools.target_free_torch import CompiledASCII95Torch, V


class HAROp(str, Enum):
    MAINTAIN = "maintain"
    AMPLIFY = "amplify"
    SUPPRESS = "suppress"
    INVERT = "invert"
    REROUTE = "reroute"


class Site(str, Enum):
    ATTENTION = "attention"
    RESIDUAL_POST_ATTENTION = "residual_post_attention"
    FFN = "ffn"
    RESIDUAL_POST_FFN = "residual_post_ffn"


@dataclass(frozen=True)
class RepresentationControl:
    """Control named representation dimensions, never anonymous tensor coordinates."""
    site: Site
    op: HAROp
    representation_ids: tuple[str, ...]
    factor: float = 1.0
    destination_ids: tuple[str, ...] = ()


class RepresentationIndex:
    def __init__(self, dimension_names: Iterable[str]):
        self.names = tuple(str(x) for x in dimension_names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("representation ids must be unique")
        self.index = {name: i for i, name in enumerate(self.names)}

    def resolve(self, ids: tuple[str, ...]) -> torch.Tensor:
        missing = [x for x in ids if x not in self.index]
        if missing:
            raise KeyError(f"unknown representation ids: {missing}")
        return torch.tensor([self.index[x] for x in ids], dtype=torch.long)


class HARController:
    """Activation control plane. It never writes model parameters or compiled tables."""
    def __init__(self, reps: RepresentationIndex, rules: Iterable[RepresentationControl] = ()):
        self.reps = reps
        self.rules = tuple(rules)

    def apply(self, site: Site, x: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
        out = x.clone()
        trace = []
        for rule in self.rules:
            if rule.site != site:
                continue
            src = self.reps.resolve(rule.representation_ids).to(out.device)
            before = out.index_select(0, src).clone()
            if rule.op == HAROp.MAINTAIN:
                pass
            elif rule.op == HAROp.AMPLIFY:
                out[src] = out[src] * rule.factor
            elif rule.op == HAROp.SUPPRESS:
                if not 0.0 <= rule.factor <= 1.0:
                    raise ValueError("suppress factor must be in [0,1]")
                out[src] = out[src] * rule.factor
            elif rule.op == HAROp.INVERT:
                out[src] = -out[src]
            elif rule.op == HAROp.REROUTE:
                dst = self.reps.resolve(rule.destination_ids).to(out.device)
                if len(dst) != len(src):
                    raise ValueError("reroute source/destination cardinality mismatch")
                moved = out[src].clone() * rule.factor
                out[src] = out[src] - moved
                out[dst] = out[dst] + moved
            after = out.index_select(0, src).clone()
            trace.append({
                "site": site.value, "op": rule.op.value,
                "representations": list(rule.representation_ids),
                "before": before.detach().cpu(), "after": after.detach().cpu(),
            })
        return out, trace


class LogitControl:
    """Separate from HAR: output-candidate admissibility only."""
    def __init__(self, allowed: torch.Tensor | None = None, suppressed: torch.Tensor | None = None):
        self.allowed = allowed
        self.suppressed = suppressed

    def apply(self, scores: torch.Tensor) -> torch.Tensor:
        out = scores.clone()
        if self.allowed is not None:
            mask = self.allowed.to(device=out.device, dtype=torch.bool)
            if mask.shape != (V,):
                raise ValueError("allowed mask must have shape (95,)")
            out = out.masked_fill(~mask, -torch.inf)
        if self.suppressed is not None:
            mask = self.suppressed.to(device=out.device, dtype=torch.bool)
            if mask.shape != (V,):
                raise ValueError("suppressed mask must have shape (95,)")
            out = out.masked_fill(mask, -torch.inf)
        return out


class ExplicitTopoKVHAR:
    """Runs the unchanged compiled equations with traceable representation control."""
    def __init__(
        self,
        model: CompiledASCII95Torch,
        dimension_names: Iterable[str],
        har: HARController | None = None,
        logits: LogitControl | None = None,
    ):
        self.model = model
        self.reps = RepresentationIndex(dimension_names)
        if len(self.reps.names) != model.d_model:
            raise ValueError("representation count must equal compiled embedding width")
        self.har = har or HARController(self.reps)
        self.logits = logits or LogitControl()

    @torch.no_grad()
    def forward_relations(self, relations, gate_mask=None):
        q, survivors, alpha, contradiction = self.model.constraint_field(relations, gate_mask)
        if contradiction:
            return {"contradiction": True, "q": q, "survivor_mask": survivors, "winner_index": None, "trace": []}

        h0 = q @ self.model.E

        # TopoKV: the compiled relation-value rows are the persistent contextual contributions.
        if relations:
            relation_values = torch.stack(
                [self.model.attention_value[oi, c] for oi, c in relations], dim=0
            )
            raw_attention_delta = alpha @ relation_values
        else:
            relation_values = torch.empty((0, self.model.d_model), dtype=self.model.E.dtype, device=self.model.E.device)
            raw_attention_delta = torch.zeros(self.model.d_model, dtype=self.model.E.dtype, device=self.model.E.device)

        attention_delta, t1 = self.har.apply(Site.ATTENTION, raw_attention_delta)
        raw_hA = h0 + attention_delta
        hA, t2 = self.har.apply(Site.RESIDUAL_POST_ATTENTION, raw_hA)

        corr = q @ self.model.ffn_corr
        raw_ffn_delta = corr * (h0 - hA)
        ffn_delta, t3 = self.har.apply(Site.FFN, raw_ffn_delta)
        raw_hF = hA + ffn_delta
        hF, t4 = self.har.apply(Site.RESIDUAL_POST_FFN, raw_hF)

        scores = -torch.sum((self.model.E - hF.unsqueeze(0)) ** 2, dim=1)
        scores = scores.masked_fill(~survivors, -torch.inf)
        scores = self.logits.apply(scores)
        winner = int(torch.argmax(scores).item()) if bool(torch.isfinite(scores).any()) else None

        return {
            "contradiction": False,
            "q": q,
            "survivor_mask": survivors,
            "attention_weights": alpha,
            "relation_values": relation_values,
            "h0": h0,
            "raw_attention_delta": raw_attention_delta,
            "attention_delta": attention_delta,
            "hA": hA,
            "raw_ffn_delta": raw_ffn_delta,
            "ffn_delta": ffn_delta,
            "hF": hF,
            "scores": scores,
            "winner_index": winner,
            "trace": t1 + t2 + t3 + t4,
        }

    def assert_weights_unchanged(self, before: dict[str, torch.Tensor]) -> None:
        now = self.model.state_dict()
        for name, tensor in before.items():
            if not torch.equal(tensor, now[name]):
                raise AssertionError(f"compiled model state mutated: {name}")
