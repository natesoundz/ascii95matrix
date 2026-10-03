from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch

from tools.target_free_torch import ASCII_MIN, ASCII_MAX, OFFSET_TO_INDEX, CompiledASCII95Torch


class Stage(str, Enum):
    RELATION_VALUE = "relation_value"
    EMBEDDING = "embedding"
    ATTENTION_DELTA = "attention_delta"
    POST_ATTENTION = "post_attention"
    FFN_DELTA = "ffn_delta"
    POST_FFN = "post_ffn"
    LOGITS = "logits"


class Op(str, Enum):
    SCALE = "scale"
    INVERT = "invert"
    SUPPRESS = "suppress"
    REROUTE = "reroute"


@dataclass(frozen=True)
class HARIntervention:
    stage: Stage
    op: Op
    dimensions: tuple[int, ...] = ()
    factor: float = 1.0
    destination_dimensions: tuple[int, ...] = ()

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        dims = self.dimensions or tuple(range(y.shape[-1]))
        idx = torch.as_tensor(dims, dtype=torch.long, device=y.device)
        if self.op == Op.SCALE:
            y[..., idx] *= self.factor
        elif self.op == Op.INVERT:
            y[..., idx] *= -1.0
        elif self.op == Op.SUPPRESS:
            y[..., idx] *= self.factor
        elif self.op == Op.REROUTE:
            if len(dims) != len(self.destination_dimensions):
                raise ValueError("reroute source/destination lengths differ")
            dst = torch.as_tensor(self.destination_dimensions, dtype=torch.long, device=y.device)
            moved = y[..., idx].clone() * self.factor
            y[..., idx] -= moved
            y[..., dst] += moved
        else:
            raise ValueError(self.op)
        return y


class CompiledHARV1:
    """HAR/TopoKV control surface for the frozen ASCII95 compiled transformer.

    This is a new adapter. It does not modify model weights or the source runtime.
    The compiled relation table is treated as the model's explicit relation-memory
    analogue of the earlier TopoKV K/V intervention surface; HAR itself operates
    on activations at named stages.
    """

    def __init__(self, model: CompiledASCII95Torch, interventions: Iterable[HARIntervention] = ()):
        self.model = model
        self.interventions = tuple(interventions)

    def _apply(self, stage: Stage, x: torch.Tensor) -> torch.Tensor:
        out = x
        for rule in self.interventions:
            if rule.stage == stage:
                out = rule.apply(out)
        return out

    @torch.no_grad()
    def forward_relations(self, relations: list[tuple[int, int]], gate_mask: torch.Tensor | None = None) -> dict:
        q, survivors, alpha, contradiction = self.model.constraint_field(relations, gate_mask)
        if contradiction:
            return {"contradiction": True, "q": q, "survivor_mask": survivors, "winner_index": None}

        h0 = self._apply(Stage.EMBEDDING, q @ self.model.E)

        if relations:
            values = torch.stack([self.model.attention_value[oi, c] for oi, c in relations], dim=0)
            values = self._apply(Stage.RELATION_VALUE, values)
            delta_a = alpha @ values
        else:
            delta_a = torch.zeros(self.model.d_model, dtype=self.model.E.dtype, device=self.model.E.device)
        delta_a = self._apply(Stage.ATTENTION_DELTA, delta_a)
        hA = self._apply(Stage.POST_ATTENTION, h0 + delta_a)

        corr = q @ self.model.ffn_corr
        delta_ffn = self._apply(Stage.FFN_DELTA, corr * (h0 - hA))
        hF = self._apply(Stage.POST_FFN, hA + delta_ffn)

        scores = -torch.sum((self.model.E - hF.unsqueeze(0)) ** 2, dim=1)
        scores = scores.masked_fill(~survivors, -torch.inf)
        scores = self._apply(Stage.LOGITS, scores)
        winner = int(torch.argmax(scores).item())
        return {
            "contradiction": False,
            "q": q,
            "survivor_mask": survivors,
            "attention_weights": alpha,
            "h0": h0,
            "attention_delta": delta_a,
            "hA": hA,
            "ffn_delta": delta_ffn,
            "hF": hF,
            "scores": scores,
            "winner_index": winner,
            "winner_ascii": winner + ASCII_MIN,
            "winner": chr(winner + ASCII_MIN),
        }

    @torch.no_grad()
    def predict_next(self, prefix: str, gate_mask: torch.Tensor | None = None) -> dict:
        if any(not ASCII_MIN <= ord(ch) <= ASCII_MAX for ch in prefix):
            raise ValueError("prefix must contain printable ASCII95 only")
        relations = []
        n = len(prefix)
        for dist in range(1, 8):
            j = n - dist
            if j < 0:
                break
            relations.append((OFFSET_TO_INDEX[-dist], ord(prefix[j]) - ASCII_MIN))
        return self.forward_relations(relations, gate_mask)
