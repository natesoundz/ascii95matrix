#!/usr/bin/env python3
"""Target-free PyTorch runtime compiled from frozen ASCII95 relational geometry.

Compile order:
    1. Load and freeze canonical compiled embedding geometry E[95, D].
    2. Compile context-only relation field from ground-truth occurrence counts.
    3. Build a target-free slot query q from visible context only.
    4. Compile additive attention values against the residual E[y] - h0.
    5. Freeze attention and replay evidence.
    6. Compile a field-fired FFN correction against the remaining residual.
    7. Verify the exact frozen runtime without supplying y at inference.

There is no optimizer, gradient update, epoch loop, random initialization, or
runtime target lookup. Ground-truth y is used only by the compiler to define
requirements and by verification to score the frozen result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import nn

ASCII_MIN = 32
ASCII_MAX = 126
V = 95
OFFSETS = np.asarray([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8)
OFFSET_TO_INDEX = {int(d): i for i, d in enumerate(OFFSETS.tolist())}
CAUSAL_OBJECTIVES = {
    "SFT_RESPONSE",
    "CHAT_ASSISTANT",
    "PROBLEM_SOLVING",
    "CODE_CAUSAL",
    "TEXT_CAUSAL",
    "CODE_INSTRUCTION",
}
BIDIRECTIONAL_OBJECTIVES = {"LM_HOLE"}
ALL_OBJECTIVES = CAUSAL_OBJECTIVES | BIDIRECTIONAL_OBJECTIVES


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ground_truth_positions(record: dict) -> list[int]:
    text = record["text"]
    positions = record.get("ground_truth_positions")
    if positions is None:
        return list(range(len(text)))
    if not isinstance(positions, list) or any(type(i) is not int for i in positions):
        raise ValueError("ground_truth_positions must be a list of integers")
    if positions != sorted(set(positions)):
        raise ValueError("ground_truth_positions must be sorted and unique")
    if any(i < 0 or i >= len(text) for i in positions):
        raise ValueError("ground_truth_positions contains an out-of-range position")
    return positions


def iter_records(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            for key in ("id", "dataset", "objective", "text"):
                if not isinstance(record.get(key), str):
                    raise ValueError(f"line {line_number}: missing string {key!r}")
            if record["objective"] not in ALL_OBJECTIVES:
                raise ValueError(f"line {line_number}: unknown objective {record['objective']!r}")
            text = record["text"]
            if not text or any(not ASCII_MIN <= ord(ch) <= ASCII_MAX for ch in text):
                raise ValueError(f"line {line_number}: text must be nonempty printable ASCII95")
            ground_truth_positions(record)
            yield record


def gate_for_position(record: dict, position: int) -> np.ndarray:
    """Return explicit candidate eligibility. Absence means no extra gate: all 95.

    Accepted optional dataset fields:
      eligible_ascii: [32, 33, ...]                       # all truth positions
      eligible_ascii_by_position: {"12": [65,66,...]}   # per position override
    """
    candidates = record.get("eligible_ascii")
    by_position = record.get("eligible_ascii_by_position")
    if by_position is not None:
        if not isinstance(by_position, dict):
            raise ValueError("eligible_ascii_by_position must be an object")
        candidates = by_position.get(str(position), by_position.get(position, candidates))

    mask = np.ones(V, dtype=bool)
    if candidates is None:
        return mask
    if not isinstance(candidates, list) or any(type(x) is not int for x in candidates):
        raise ValueError("eligible candidate list must contain integer ASCII codes")
    if any(x < ASCII_MIN or x > ASCII_MAX for x in candidates):
        raise ValueError("eligible candidate ASCII code outside 32..126")
    mask[:] = False
    if candidates:
        mask[np.asarray(candidates, dtype=np.int64) - ASCII_MIN] = True
    return mask


def visible_relations(record: dict, position: int) -> list[tuple[int, int]]:
    text = record["text"]
    obj = record["objective"]
    rels: list[tuple[int, int]] = []
    for d in OFFSETS.tolist():
        d = int(d)
        if obj in CAUSAL_OBJECTIVES and d > 0:
            continue
        j = position + d
        if 0 <= j < len(text):
            rels.append((OFFSET_TO_INDEX[d], ord(text[j]) - ASCII_MIN))
    return rels


@dataclass(frozen=True)
class FrozenGeometry:
    E: np.ndarray
    ascii_code: np.ndarray
    dimension_name: np.ndarray
    source_geometry: Path
    source_sha256: str

    @classmethod
    def load(cls, directory: str | Path) -> "FrozenGeometry":
        directory = Path(directory).resolve()
        geometry = directory / "geometry.npz"
        manifest = directory / "frozen_manifest.json"
        if not geometry.exists() or not manifest.exists():
            raise FileNotFoundError("expected geometry.npz and frozen_manifest.json")
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        actual = sha256_file(geometry)
        expected = meta.get("geometry_sha256")
        if expected != actual:
            raise ValueError("frozen geometry hash mismatch")
        with np.load(geometry, allow_pickle=False) as z:
            required = {"E", "ascii_code", "dimension_name"}
            missing = required - set(z.files)
            if missing:
                raise ValueError(f"geometry missing arrays {sorted(missing)}")
            E = np.asarray(z["E"], dtype=np.float64)
            ascii_code = np.asarray(z["ascii_code"], dtype=np.int16)
            dimension_name = np.asarray(z["dimension_name"]).astype(str)
        if E.ndim != 2 or E.shape[0] != V:
            raise ValueError(f"E must have shape (95,D), got {E.shape}")
        if not np.array_equal(ascii_code, np.arange(ASCII_MIN, ASCII_MAX + 1, dtype=np.int16)):
            raise ValueError("ascii_code must be exactly 32..126")
        if dimension_name.shape != (E.shape[1],):
            raise ValueError("dimension_name length must equal E width")
        if not np.all(np.isfinite(E)):
            raise ValueError("E contains non-finite values")
        return cls(E=E, ascii_code=ascii_code, dimension_name=dimension_name,
                   source_geometry=geometry, source_sha256=actual)


def compile_counts(dataset: str | Path) -> np.ndarray:
    counts = np.zeros((len(OFFSETS), V, V), dtype=np.float64)
    for record in iter_records(dataset):
        text = record["text"]
        for p in ground_truth_positions(record):
            y = ord(text[p]) - ASCII_MIN
            gate = gate_for_position(record, p)
            if not gate[y]:
                raise ValueError(
                    f"{record['dataset']}/{record['id']} position {p}: gate excludes ground truth {text[p]!r}"
                )
            for oi, c in visible_relations(record, p):
                counts[oi, c, y] += 1.0
    if counts.sum() <= 0:
        raise ValueError("dataset produced no visible context relations")
    return counts


def relation_specificity(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context_totals = counts.sum(axis=2)
    target_totals = counts.sum(axis=1)
    forward = np.divide(
        counts, context_totals[:, :, None], out=np.zeros_like(counts),
        where=context_totals[:, :, None] > 0,
    )
    reverse = np.divide(
        counts, target_totals[:, None, :], out=np.zeros_like(counts),
        where=target_totals[:, None, :] > 0,
    )
    mutual = np.sqrt(forward * reverse)
    return forward, reverse, mutual


def field_from_relations(
    mutual: np.ndarray,
    relations: list[tuple[int, int]],
    gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Construct target-free candidate field q and attention relation weights.

    Survivor semantics are strict: a candidate must pass the supplied gate and
    have nonzero compiled relation support for every visible relation. No silent
    fallback is used when the intersection is empty.
    """
    gate = np.asarray(gate, dtype=bool)
    if gate.shape != (V,):
        raise ValueError("gate must have shape (95,)")
    survivors = gate.copy()
    if not relations:
        if not np.any(survivors):
            return np.zeros(V), survivors, np.empty(0), True
        q = survivors.astype(np.float64)
        q /= q.sum()
        return q, survivors, np.empty(0), False

    for oi, c in relations:
        survivors &= mutual[oi, c] > 0.0
    if not np.any(survivors):
        return np.zeros(V), survivors, np.zeros(len(relations)), True

    idx = np.flatnonzero(survivors)
    log_field = np.zeros(len(idx), dtype=np.float64)
    relation_logs: list[np.ndarray] = []
    for oi, c in relations:
        vals = np.log(mutual[oi, c, idx])
        relation_logs.append(vals)
        log_field += vals
    log_field /= float(len(relations))
    log_field -= np.max(log_field)
    p = np.exp(log_field)
    p /= p.sum()
    q = np.zeros(V, dtype=np.float64)
    q[idx] = p

    scores = np.asarray([float(np.dot(p, v)) for v in relation_logs], dtype=np.float64)
    scores -= np.max(scores)
    alpha = np.exp(scores)
    alpha /= alpha.sum()
    return q, survivors, alpha, False


def relation_id(oi: int, c: int) -> int:
    return int(oi) * V + int(c)


def compile_attention_values(
    dataset: str | Path,
    E: np.ndarray,
    mutual: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Least-norm compile of additive relation values after E is frozen."""
    R = len(OFFSETS) * V
    D = E.shape[1]
    used = np.zeros(R, dtype=bool)
    occurrence_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    contradictions = 0

    for record in iter_records(dataset):
        text = record["text"]
        for p in ground_truth_positions(record):
            y = ord(text[p]) - ASCII_MIN
            rels = visible_relations(record, p)
            gate = gate_for_position(record, p)
            q, survivors, alpha, contradiction = field_from_relations(mutual, rels, gate)
            if contradiction:
                contradictions += 1
                continue
            if not survivors[y]:
                raise ValueError(
                    f"compiled field eliminated its own ground truth at {record['id']}:{p}"
                )
            h0 = q @ E
            target_residual = E[y] - h0
            rids = np.asarray([relation_id(oi, c) for oi, c in rels], dtype=np.int64)
            if len(rids):
                used[rids] = True
            occurrence_cache.append((rids, alpha, target_residual, y))

    active = np.flatnonzero(used)
    if active.size == 0:
        raise ValueError("no observed attention relations")
    local = {int(rid): i for i, rid in enumerate(active.tolist())}
    M = len(active)
    ATA = np.zeros((M, M), dtype=np.float64)
    ATR = np.zeros((M, D), dtype=np.float64)

    for rids, alpha, residual, _ in occurrence_cache:
        if not len(rids):
            continue
        ii = np.asarray([local[int(r)] for r in rids], dtype=np.int64)
        ATA[np.ix_(ii, ii)] += np.outer(alpha, alpha)
        ATR[ii] += alpha[:, None] * residual[None, :]

    # Direct least-norm normal-equation solution. Zero values are an admissible
    # solution, so the global post-attention squared residual cannot be worse
    # than the frozen-embedding starting point except for numerical tolerance.
    Vactive = np.linalg.pinv(ATA, rcond=1e-12) @ ATR
    values = np.zeros((R, D), dtype=np.float64)
    values[active] = Vactive

    before2 = 0.0
    after2 = 0.0
    coords = 0
    for rids, alpha, residual, _ in occurrence_cache:
        delta = np.zeros(D, dtype=np.float64)
        if len(rids):
            delta = alpha @ values[rids]
        before2 += float(np.dot(residual, residual))
        remaining = residual - delta
        after2 += float(np.dot(remaining, remaining))
        coords += D

    audit = {
        "compiled_relations": int(M),
        "occurrences": int(len(occurrence_cache)),
        "contradictions": int(contradictions),
        "pre_attention_residual_rms": math.sqrt(before2 / max(1, coords)),
        "post_attention_residual_rms": math.sqrt(after2 / max(1, coords)),
        "normal_equation_rank": int(np.linalg.matrix_rank(ATA, tol=1e-10)),
    }
    return values.reshape(len(OFFSETS), V, D), audit


def replay_states(
    dataset: str | Path,
    E: np.ndarray,
    mutual: np.ndarray,
    attention_value: np.ndarray,
) -> list[dict]:
    out: list[dict] = []
    for record in iter_records(dataset):
        text = record["text"]
        for p in ground_truth_positions(record):
            y = ord(text[p]) - ASCII_MIN
            rels = visible_relations(record, p)
            gate = gate_for_position(record, p)
            q, survivors, alpha, contradiction = field_from_relations(mutual, rels, gate)
            if contradiction:
                out.append({"contradiction": True, "y": y})
                continue
            h0 = q @ E
            if rels:
                Vrows = np.asarray([attention_value[oi, c] for oi, c in rels], dtype=np.float64)
                delta_a = alpha @ Vrows
            else:
                delta_a = np.zeros(E.shape[1], dtype=np.float64)
            hA = h0 + delta_a
            out.append({
                "contradiction": False,
                "y": y,
                "q": q,
                "survivors": survivors,
                "h0": h0,
                "hA": hA,
            })
    return out


def compile_field_ffn(states: list[dict], E: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compile remaining FFN residual after frozen attention.

    Two tables are produced:
      truth_corr[y,k]  - the historical per-target scalar normal equation,
                         retained as an audit of the old one-hot gate form.
      field_corr[y,k]  - the target-free field-fired normal-equation extension
                         actually used by runtime. When q is one-hot, this
                         collapses to the scalar form.

    Runtime equation:
        d = h0 - hA
        corr_field[k] = sum_y q[y] * field_corr[y,k]
        delta_ffn[k] = d[k] * corr_field[k]
    """
    valid = [s for s in states if not s["contradiction"]]
    if not valid:
        raise ValueError("no valid states for FFN compilation")
    Q = np.stack([s["q"] for s in valid], axis=0)             # N,V
    H0 = np.stack([s["h0"] for s in valid], axis=0)          # N,D
    HA = np.stack([s["hA"] for s in valid], axis=0)          # N,D
    Y = np.asarray([s["y"] for s in valid], dtype=np.int64)  # N
    target = E[Y]
    d = H0 - HA
    rem = target - HA
    N, D = d.shape

    # Old one-hot-gated scalar normal equation, for audit/continuity.
    num = np.zeros((V, D), dtype=np.float64)
    den = np.zeros((V, D), dtype=np.float64)
    np.add.at(num, Y, d * rem)
    np.add.at(den, Y, d * d)
    truth_corr = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-18)

    # Target-free field-fired direct solve. For each coordinate k:
    # rem[:,k] ~= (Q @ C[:,k]) * d[:,k]
    # This is the exact linear normal-equation extension of the scalar formula.
    field_corr = np.zeros((V, D), dtype=np.float64)
    for k in range(D):
        x = Q * d[:, k:k+1]
        if not np.any(np.abs(x) > 0):
            continue
        field_corr[:, k] = np.linalg.pinv(x, rcond=1e-12) @ rem[:, k]

    corr_runtime = Q @ field_corr
    delta = d * corr_runtime
    hf = HA + delta

    emb2 = np.sum((target - H0) ** 2)
    att2 = np.sum((target - HA) ** 2)
    ffn2 = np.sum((target - hf) ** 2)
    coords = N * D
    audit = {
        "occurrences": int(N),
        "embedding_residual_rms": math.sqrt(float(emb2) / max(1, coords)),
        "post_attention_residual_rms": math.sqrt(float(att2) / max(1, coords)),
        "post_ffn_residual_rms": math.sqrt(float(ffn2) / max(1, coords)),
        "field_corr_nonzero": int(np.count_nonzero(np.abs(field_corr) > 0.0)),
        "truth_corr_nonzero": int(np.count_nonzero(np.abs(truth_corr) > 0.0)),
    }
    return field_corr, truth_corr, audit


class CompiledASCII95Torch(nn.Module):
    """Frozen target-free PyTorch execution graph."""

    def __init__(
        self,
        embedding: torch.Tensor,
        mutual: torch.Tensor,
        attention_value: torch.Tensor,
        ffn_corr: torch.Tensor,
    ) -> None:
        super().__init__()
        if embedding.ndim != 2 or embedding.shape[0] != V:
            raise ValueError("embedding must have shape (95,D)")
        dtype = embedding.dtype
        self.embedding = nn.Embedding.from_pretrained(embedding.clone(), freeze=True)
        self.register_buffer("mutual", mutual.to(dtype=dtype).clone())
        self.register_buffer("attention_value", attention_value.to(dtype=dtype).clone())
        self.register_buffer("ffn_corr", ffn_corr.to(dtype=dtype).clone())
        self.d_model = int(embedding.shape[1])

    @property
    def E(self) -> torch.Tensor:
        return self.embedding.weight

    def constraint_field(
        self,
        relations: list[tuple[int, int]],
        gate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        device = self.E.device
        dtype = self.E.dtype
        if gate_mask is None:
            survivors = torch.ones(V, dtype=torch.bool, device=device)
        else:
            survivors = gate_mask.to(device=device, dtype=torch.bool).clone()
            if survivors.shape != (V,):
                raise ValueError("gate_mask must have shape (95,)")
        if not relations:
            if not bool(survivors.any()):
                return torch.zeros(V, dtype=dtype, device=device), survivors, torch.empty(0, dtype=dtype, device=device), True
            q = survivors.to(dtype)
            q = q / q.sum()
            return q, survivors, torch.empty(0, dtype=dtype, device=device), False

        logs = []
        for oi, c in relations:
            r = self.mutual[oi, c]
            survivors &= r > 0
        if not bool(survivors.any()):
            return torch.zeros(V, dtype=dtype, device=device), survivors, torch.zeros(len(relations), dtype=dtype, device=device), True
        idx = torch.nonzero(survivors, as_tuple=False).flatten()
        for oi, c in relations:
            logs.append(torch.log(self.mutual[oi, c, idx]))
        L = torch.stack(logs, dim=0)
        q_local = torch.softmax(L.mean(dim=0), dim=0)
        q = torch.zeros(V, dtype=dtype, device=device)
        q[idx] = q_local
        scores = L @ q_local
        alpha = torch.softmax(scores, dim=0)
        return q, survivors, alpha, False

    @torch.no_grad()
    def forward_relations(
        self,
        relations: list[tuple[int, int]],
        gate_mask: torch.Tensor | None = None,
    ) -> dict:
        q, survivors, alpha, contradiction = self.constraint_field(relations, gate_mask)
        if contradiction:
            return {
                "contradiction": True,
                "q": q,
                "survivor_mask": survivors,
                "winner_index": None,
                "scores": torch.full((V,), -torch.inf, dtype=self.E.dtype, device=self.E.device),
            }
        h0 = q @ self.E
        if relations:
            values = torch.stack([self.attention_value[oi, c] for oi, c in relations], dim=0)
            delta_a = alpha @ values
        else:
            delta_a = torch.zeros(self.d_model, dtype=self.E.dtype, device=self.E.device)
        hA = h0 + delta_a
        corr = q @ self.ffn_corr
        delta_ffn = corr * (h0 - hA)
        hF = hA + delta_ffn
        scores = -torch.sum((self.E - hF.unsqueeze(0)) ** 2, dim=1)
        scores = scores.masked_fill(~survivors, -torch.inf)
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
        }

    @torch.no_grad()
    def predict_next(self, prefix: str, gate_mask: torch.Tensor | None = None) -> dict:
        if any(not ASCII_MIN <= ord(ch) <= ASCII_MAX for ch in prefix):
            raise ValueError("prefix must contain printable ASCII95 only")
        relations: list[tuple[int, int]] = []
        n = len(prefix)
        for dist in range(1, 8):
            j = n - dist
            if j < 0:
                break
            d = -dist
            relations.append((OFFSET_TO_INDEX[d], ord(prefix[j]) - ASCII_MIN))
        result = self.forward_relations(relations, gate_mask)
        if result["winner_index"] is not None:
            result["winner_ascii"] = result["winner_index"] + ASCII_MIN
            result["winner"] = chr(result["winner_ascii"])
        else:
            result["winner_ascii"] = None
            result["winner"] = None
        result["survivors"] = [
            chr(i + ASCII_MIN) for i in torch.nonzero(result["survivor_mask"], as_tuple=False).flatten().tolist()
        ]
        result["eliminated"] = [
            chr(i + ASCII_MIN) for i in torch.nonzero(~result["survivor_mask"], as_tuple=False).flatten().tolist()
        ]
        return result


def top1_accuracy(states: list[dict], E: np.ndarray, ffn_corr: np.ndarray | None = None) -> dict:
    counts = {"h0": 0, "hA": 0, "hF": 0, "n": 0}
    for s in states:
        if s["contradiction"]:
            continue
        y = int(s["y"])
        survivors = s["survivors"]
        for name in ("h0", "hA"):
            h = s[name]
            score = -np.sum((E - h[None, :]) ** 2, axis=1)
            score[~survivors] = -np.inf
            counts[name] += int(np.argmax(score) == y)
        if ffn_corr is not None:
            q, h0, hA = s["q"], s["h0"], s["hA"]
            hF = hA + (q @ ffn_corr) * (h0 - hA)
            score = -np.sum((E - hF[None, :]) ** 2, axis=1)
            score[~survivors] = -np.inf
            counts["hF"] += int(np.argmax(score) == y)
        counts["n"] += 1
    n = max(1, counts["n"])
    return {
        "occurrences": counts["n"],
        "h0_top1": counts["h0"] / n,
        "hA_top1": counts["hA"] / n,
        "hF_top1": counts["hF"] / n if ffn_corr is not None else None,
    }


def compile_runtime(dataset: str | Path, geometry_dir: str | Path, out_dir: str | Path) -> dict:
    dataset = Path(dataset).resolve()
    geometry = FrozenGeometry.load(geometry_dir)
    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("output directory must be empty")
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = out_dir / "evidence.jsonl"
    shutil.copyfile(dataset, snapshot)
    source_hash = sha256_file(snapshot)

    counts = compile_counts(snapshot)
    forward, reverse, mutual = relation_specificity(counts)

    attention_value, attention_audit = compile_attention_values(snapshot, geometry.E, mutual)
    states = replay_states(snapshot, geometry.E, mutual, attention_value)
    field_corr, truth_corr, residual_audit = compile_field_ffn(states, geometry.E)
    accuracy = top1_accuracy(states, geometry.E, field_corr)

    if sha256_file(snapshot) != source_hash:
        raise ValueError("evidence changed during compilation")

    passed = (
        attention_audit["contradictions"] == 0
        and residual_audit["post_attention_residual_rms"] <= residual_audit["embedding_residual_rms"] + 1e-10
        and residual_audit["post_ffn_residual_rms"] <= residual_audit["post_attention_residual_rms"] + 1e-10
        and np.all(np.isfinite(attention_value))
        and np.all(np.isfinite(field_corr))
    )

    artifact = {
        "format": "ascii95-target-free-torch-v1",
        "embedding": torch.from_numpy(geometry.E.copy()),
        "mutual": torch.from_numpy(mutual.copy()),
        "attention_value": torch.from_numpy(attention_value.copy()),
        "ffn_corr": torch.from_numpy(field_corr.copy()),
        "truth_gated_ffn_corr_audit": torch.from_numpy(truth_corr.copy()),
        "counts": torch.from_numpy(counts.copy()),
        "offsets": torch.from_numpy(OFFSETS.copy()),
        "ascii_min": ASCII_MIN,
        "ascii_max": ASCII_MAX,
        "source_geometry_sha256": geometry.source_sha256,
        "source_evidence_sha256": source_hash,
        "target_free_runtime": True,
    }
    artifact_path = out_dir / "runtime.pt"
    torch.save(artifact, artifact_path)

    # Reload exact serialized tensors and execute a no-target inference smoke test.
    loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
    model = CompiledASCII95Torch(
        loaded["embedding"], loaded["mutual"], loaded["attention_value"], loaded["ffn_corr"]
    ).eval()
    smoke = model.predict_next("The dog ")
    serialization_exact = (
        torch.equal(loaded["embedding"], artifact["embedding"])
        and torch.equal(loaded["mutual"], artifact["mutual"])
        and torch.equal(loaded["attention_value"], artifact["attention_value"])
        and torch.equal(loaded["ffn_corr"], artifact["ffn_corr"])
    )
    passed = bool(passed and serialization_exact)

    report = {
        "status": "PASS" if passed else "FAIL",
        "compile_order": [
            "EMBEDDING_LOADED_FROZEN",
            "CONTEXT_RELATIONS_COMPILED",
            "ATTENTION_COMPILED_FROZEN",
            "FFN_COMPILED_FROM_REMAINING_RESIDUAL",
            "TARGET_FREE_RUNTIME_VERIFIED",
        ],
        "dimensions": {"vocab": V, "d_model": int(geometry.E.shape[1]), "offsets": OFFSETS.tolist()},
        "geometry_sha256": geometry.source_sha256,
        "evidence_sha256": source_hash,
        "runtime_sha256": sha256_file(artifact_path),
        "attention": attention_audit,
        "residual": residual_audit,
        "top1_replay": accuracy,
        "serialization_exact": serialization_exact,
        "runtime_target_input": False,
        "runtime_smoke": {
            "prefix": "The dog ",
            "winner": smoke["winner"],
            "survivor_count": len(smoke["survivors"]),
            "contradiction": smoke["contradiction"],
        },
        "ffn_note": (
            "truth_gated_ffn_corr_audit preserves the old per-target scalar normal equation; "
            "ffn_corr is its field-fired normal-equation extension and is the only FFN table used at runtime."
        ),
    }
    (out_dir / "compile_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("verification failed; inspect compile_report.json")
    return report


def load_runtime(path: str | Path, device: str = "cpu") -> CompiledASCII95Torch:
    z = torch.load(Path(path), map_location=device, weights_only=True)
    required = {"embedding", "mutual", "attention_value", "ffn_corr"}
    missing = required - set(z)
    if missing:
        raise ValueError(f"runtime artifact missing {sorted(missing)}")
    return CompiledASCII95Torch(z["embedding"], z["mutual"], z["attention_value"], z["ffn_corr"]).to(device).eval()


def self_test() -> dict:
    rng = np.random.default_rng(0)
    D = 16
    # Unique deterministic synthetic geometry only for implementation integrity.
    E = rng.normal(size=(V, D)).astype(np.float64)
    counts = np.zeros((len(OFFSETS), V, V), dtype=np.float64)
    # A compact deterministic transition system with two causal offsets.
    for c in range(V):
        y1 = (c + 1) % V
        y2 = (c + 7) % V
        counts[OFFSET_TO_INDEX[-1], c, y1] += 8
        counts[OFFSET_TO_INDEX[-1], c, y2] += 1
        counts[OFFSET_TO_INDEX[-2], c, (c + 2) % V] += 4
        counts[OFFSET_TO_INDEX[-2], c, (c + 9) % V] += 1
    _, _, mutual = relation_specificity(counts)
    # Target-free construction must not need y.
    rels = [(OFFSET_TO_INDEX[-2], 10), (OFFSET_TO_INDEX[-1], 11)]
    q, survivors, alpha, contradiction = field_from_relations(mutual, rels, np.ones(V, dtype=bool))
    if contradiction:
        # These arbitrary relations need not intersect; use one relation for the integrity path.
        rels = [(OFFSET_TO_INDEX[-1], 11)]
        q, survivors, alpha, contradiction = field_from_relations(mutual, rels, np.ones(V, dtype=bool))
    if contradiction or not np.isclose(q.sum(), 1.0):
        raise AssertionError("constraint field failed")
    attention = np.zeros((len(OFFSETS), V, D), dtype=np.float64)
    corr = np.zeros((V, D), dtype=np.float64)
    model = CompiledASCII95Torch(
        torch.from_numpy(E), torch.from_numpy(mutual), torch.from_numpy(attention), torch.from_numpy(corr)
    ).eval()
    result = model.forward_relations(rels)
    if result["contradiction"] or result["hF"].shape != (D,):
        raise AssertionError("PyTorch execution graph failed")
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("runtime contains trainable parameters")
    return {
        "passed": True,
        "embedding_shape": list(model.E.shape),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "target_argument_in_runtime": False,
        "survivor_count": int(result["survivor_mask"].sum().item()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--dataset", required=True)
    c.add_argument("--geometry-dir", required=True)
    c.add_argument("--out", required=True)
    v = sub.add_parser("predict")
    v.add_argument("--runtime", required=True)
    v.add_argument("--text", required=True)
    v.add_argument("--device", default="cpu")
    sub.add_parser("self-test")
    args = p.parse_args()
    try:
        if args.command == "compile":
            print(json.dumps(compile_runtime(args.dataset, args.geometry_dir, args.out), indent=2))
        elif args.command == "predict":
            model = load_runtime(args.runtime, args.device)
            r = model.predict_next(args.text)
            printable = {
                "contradiction": r["contradiction"],
                "winner": r["winner"],
                "winner_ascii": r["winner_ascii"],
                "survivors": r["survivors"],
                "eliminated": r["eliminated"],
            }
            print(json.dumps(printable, indent=2))
        else:
            print(json.dumps(self_test(), indent=2))
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())