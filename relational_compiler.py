#!/usr/bin/env python3
"""
relational_compiler.py

Deterministic compiler for a frozen ASCII95 x 437 relational geometry.

Compilation boundary:

CANONICAL 95 x 437 RELATIONSHIP REGISTRY
                    |
                    v
        ALL VALID OCCURRENCE EVIDENCE
                    |
                    v
          RELATIONAL MEASUREMENTS
                 g[i,k]
                    |
                    v
        CHARACTER EVIDENCE FIELD
             G[95 x 437]
                    |
          +---------+---------+
          |         |         |
          v         v         v
         H[k]    rho[c,k]    mu[k]
                              |
                              v
                           sigma[k]
                              |
                              v
                    TOKEN GEOMETRY
                      E[95 x 437]
                              |
                           FREEZE

The registry defines measurable relationships / participation only.
It does NOT provide embedding coordinates.

No Q/K/V or FFN compilation law is invented here. Those stages are
blocked behind the validated frozen geometry boundary.

Requirements:
    Python >= 3.10
    numpy >= 1.24

Registry NPZ:
    ascii_code       shape (95,)      exactly 32..126
    dimension_name   shape (437,)
    participation    shape (95,437)   bool / 0-1

Evidence NPZ:
    y                shape (N,)       observed correct ASCII code
    g                shape (N,437)    relational measurements
    w                shape (N,)       nonnegative evidence weight
    source_id        shape (N,)       optional trace identifier

For g:
    finite value = valid measurement
    NaN          = unavailable / invalid measurement
    +/-inf       = illegal

Compiled quantities:
    G[c,k]       = sum_i:y_i=c w_i * g[i,k]

    H[k]         = sum_c phi(G[c,k])

    lambda[k]    = 1 / H[k]

    rho[c,k]     = lambda[k] * phi(G[c,k])

    mu[k]        = sum_c w_ck[c,k] G[c,k] / sum_c w_ck[c,k]

    sigma[k]     = sqrt(
                        sum_c w_ck[c,k] (G[c,k]-mu[k])^2
                        ----------------------------------
                              sum_c w_ck[c,k]
                      )

    E[c,k]       = (G[c,k]-mu[k]) / (sigma[k] + epsilon)

Trace invariant:
    E[c,k]
      -> sigma[k], mu[k]
      -> G[c,k]
      -> g[i,k]
      -> source occurrence evidence
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
from typing import Iterable

import numpy as np


ASCII_MIN = 32
ASCII_MAX = 126
ASCII_COUNT = 95
DIMENSION_COUNT = 437

ASCII_CODES = np.arange(
    ASCII_MIN,
    ASCII_MAX + 1,
    dtype=np.int16,
)

PHI_MODES = (
    "abs",
    "square",
    "identity_nonnegative",
)

CENTER_WEIGHT_MODES = (
    "uniform",
    "occurrence_weight_mass",
    "valid_occurrence_count",
    "external",
)

STATE_MAPPERS = (
    "field_zscore",
)

CORRESPONDENCE_MODES = (
    "negative_squared_distance",
    "dot",
    "cosine",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: str | Path, obj: dict) -> None:
    Path(path).write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ensure_shape(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")


def ensure_finite(name: str, arr: np.ndarray) -> None:
    mask = ~np.isfinite(arr)
    if np.any(mask):
        locations = np.argwhere(mask)[:20].tolist()
        raise ValueError(f"{name} contains non-finite values at {locations}")


def character_index(ascii_code: int) -> int:
    if ascii_code < ASCII_MIN or ascii_code > ASCII_MAX:
        raise ValueError(
            f"ASCII code must be {ASCII_MIN}..{ASCII_MAX}, got {ascii_code}"
        )
    return ascii_code - ASCII_MIN


@dataclass(frozen=True)
class RelationshipRegistry:
    ascii_code: np.ndarray
    dimension_name: np.ndarray
    participation: np.ndarray
    source_path: Path
    source_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "RelationshipRegistry":
        path = Path(path).resolve()

        with np.load(path, allow_pickle=False) as data:
            required = {"ascii_code", "dimension_name", "participation"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"registry missing arrays: {sorted(missing)}")

            ascii_code = np.asarray(data["ascii_code"], dtype=np.int16)
            dimension_name = np.asarray(data["dimension_name"]).astype(str)
            participation_raw = np.asarray(data["participation"])

        ensure_shape("registry.ascii_code", ascii_code, (ASCII_COUNT,))
        ensure_shape(
            "registry.dimension_name",
            dimension_name,
            (DIMENSION_COUNT,),
        )
        ensure_shape(
            "registry.participation",
            participation_raw,
            (ASCII_COUNT, DIMENSION_COUNT),
        )

        if not np.array_equal(ascii_code, ASCII_CODES):
            raise ValueError(
                "registry.ascii_code must be exactly the printable "
                "ASCII sequence 32..126 in ascending order"
            )

        if any(not name.strip() for name in dimension_name.tolist()):
            raise ValueError("all 437 dimension names must be non-empty")

        if len(set(dimension_name.tolist())) != DIMENSION_COUNT:
            raise ValueError("all 437 dimension names must be unique")

        if participation_raw.dtype == np.bool_:
            participation = participation_raw.copy()
        else:
            unique = np.unique(participation_raw)
            if not np.all(np.isin(unique, [0, 1])):
                raise ValueError(
                    "registry.participation must contain only 0/1 "
                    "or boolean values"
                )
            participation = participation_raw.astype(bool)

        return cls(
            ascii_code=ascii_code,
            dimension_name=dimension_name,
            participation=participation,
            source_path=path,
            source_sha256=sha256_file(path),
        )


@dataclass(frozen=True)
class EvidenceShard:
    path: Path
    sha256: str
    y: np.ndarray
    g: np.ndarray
    w: np.ndarray
    source_id: np.ndarray

    @property
    def occurrence_count(self) -> int:
        return int(self.y.shape[0])

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceShard":
        path = Path(path).resolve()

        with np.load(path, allow_pickle=False) as data:
            required = {"y", "g", "w"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing arrays {sorted(missing)}")

            y = np.asarray(data["y"], dtype=np.int16)
            g = np.asarray(data["g"], dtype=np.float64)
            w = np.asarray(data["w"], dtype=np.float64)

            if "source_id" in data.files:
                source_id = np.asarray(data["source_id"]).astype(str)
            else:
                source_id = np.asarray(
                    [f"{path.name}:{i}" for i in range(y.shape[0])],
                    dtype=str,
                )

        if y.ndim != 1:
            raise ValueError(f"{path}: y must be rank-1")

        n = y.shape[0]

        ensure_shape(f"{path}.g", g, (n, DIMENSION_COUNT))
        ensure_shape(f"{path}.w", w, (n,))
        ensure_shape(f"{path}.source_id", source_id, (n,))

        if np.any((y < ASCII_MIN) | (y > ASCII_MAX)):
            bad = np.where((y < ASCII_MIN) | (y > ASCII_MAX))[0][:20]
            raise ValueError(
                f"{path}: y contains non-printable-ASCII "
                f"targets at rows {bad.tolist()}"
            )

        if np.any(np.isinf(g)):
            bad = np.argwhere(np.isinf(g))[:20].tolist()
            raise ValueError(
                f"{path}: g contains infinity at {bad}; "
                f"use NaN for unavailable measurements"
            )

        if not np.all(np.isfinite(w)):
            bad = np.where(~np.isfinite(w))[0][:20].tolist()
            raise ValueError(
                f"{path}: w contains non-finite values at rows {bad}"
            )

        if np.any(w < 0.0):
            bad = np.where(w < 0.0)[0][:20].tolist()
            raise ValueError(
                f"{path}: evidence weights must be nonnegative; "
                f"negative rows {bad}"
            )

        return cls(
            path=path,
            sha256=sha256_file(path),
            y=y,
            g=g,
            w=w,
            source_id=source_id,
        )


def phi(G: np.ndarray, mode: str) -> np.ndarray:
    if mode == "abs":
        return np.abs(G)

    if mode == "square":
        return np.square(G)

    if mode == "identity_nonnegative":
        if np.any(G < 0.0):
            location = np.argwhere(G < 0.0)[0].tolist()
            raise ValueError(
                "phi='identity_nonnegative' requires G >= 0 "
                f"everywhere; negative value at {location}"
            )
        return G.copy()

    raise ValueError(
        f"unknown phi mode {mode!r}; expected one of {PHI_MODES}"
    )


def compile_character_evidence_field(
    registry: RelationshipRegistry,
    shards: Iterable[EvidenceShard],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    G = np.zeros(
        (ASCII_COUNT, DIMENSION_COUNT),
        dtype=np.float64,
    )

    occurrence_weight_mass = np.zeros_like(G)

    valid_occurrence_count = np.zeros(
        (ASCII_COUNT, DIMENSION_COUNT),
        dtype=np.int64,
    )

    for shard in shards:
        for row in range(shard.occurrence_count):
            c = int(shard.y[row]) - ASCII_MIN
            measurements = shard.g[row]
            valid = np.isfinite(measurements)

            illegal = valid & ~registry.participation[c]

            if np.any(illegal):
                dimensions = np.where(illegal)[0][:20].tolist()
                raise ValueError(
                    f"{shard.path}: row {row}, ASCII "
                    f"{int(shard.y[row])} contains measurements "
                    f"for non-participating dimensions {dimensions}"
                )

            if not np.any(valid):
                continue

            weight = float(shard.w[row])

            G[c, valid] += weight * measurements[valid]
            occurrence_weight_mass[c, valid] += weight
            valid_occurrence_count[c, valid] += 1

    return G, occurrence_weight_mass, valid_occurrence_count


def build_center_weights(
    mode: str,
    occurrence_weight_mass: np.ndarray,
    valid_occurrence_count: np.ndarray,
    external_path: str | Path | None,
) -> np.ndarray:

    if mode == "uniform":
        return np.ones(
            (ASCII_COUNT, DIMENSION_COUNT),
            dtype=np.float64,
        )

    if mode == "occurrence_weight_mass":
        weights = occurrence_weight_mass.astype(np.float64, copy=True)

    elif mode == "valid_occurrence_count":
        weights = valid_occurrence_count.astype(np.float64, copy=True)

    elif mode == "external":
        if external_path is None:
            raise ValueError(
                "center-weight='external' requires --center-weight-npz"
            )

        with np.load(external_path, allow_pickle=False) as data:
            if "w_ck" not in data.files:
                raise ValueError(
                    "external center-weight NPZ must contain w_ck"
                )

            weights = np.asarray(data["w_ck"], dtype=np.float64)

        ensure_shape(
            "external w_ck",
            weights,
            (ASCII_COUNT, DIMENSION_COUNT),
        )

    else:
        raise ValueError(
            f"unknown center-weight mode {mode!r}; "
            f"expected one of {CENTER_WEIGHT_MODES}"
        )

    ensure_finite("center weights", weights)

    if np.any(weights < 0.0):
        location = np.argwhere(weights < 0.0)[0].tolist()
        raise ValueError(
            f"center weights must be nonnegative; negative value at {location}"
        )

    denominator = np.sum(weights, axis=0)

    missing = np.where(denominator <= 0.0)[0]

    if missing.size:
        raise ValueError(
            "center weights contain dimensions with zero total "
            f"weight: {missing[:50].tolist()}"
        )

    return weights


@dataclass(frozen=True)
class CompileConfig:
    phi_mode: str
    center_weight_mode: str
    epsilon: float = 1e-12
    center_weight_npz: str | Path | None = None

    def validate(self) -> None:
        if self.phi_mode not in PHI_MODES:
            raise ValueError(f"phi_mode must be one of {PHI_MODES}")

        if self.center_weight_mode not in CENTER_WEIGHT_MODES:
            raise ValueError(
                "center_weight_mode must be one of "
                f"{CENTER_WEIGHT_MODES}"
            )

        if not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite")

        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")


def compile_geometry(
    registry_path: str | Path,
    evidence_paths: Iterable[str | Path],
    output_directory: str | Path,
    config: CompileConfig,
) -> Path:

    config.validate()

    registry = RelationshipRegistry.load(registry_path)

    evidence_paths = list(evidence_paths)

    if not evidence_paths:
        raise ValueError("at least one evidence shard is required")

    shards = [EvidenceShard.load(path) for path in evidence_paths]

    (
        G,
        occurrence_weight_mass,
        valid_occurrence_count,
    ) = compile_character_evidence_field(
        registry,
        shards,
    )

    ensure_finite("G", G)

    phi_G = phi(G, config.phi_mode)

    H = np.sum(phi_G, axis=0)

    zero_H = np.where(H <= 0.0)[0]

    if zero_H.size:
        raise ValueError(
            "cannot construct reciprocal whole-field reference: "
            "H[k] is zero/nonpositive for dimensions "
            f"{zero_H[:50].tolist()}"
        )

    lambda_ = 1.0 / H

    rho = phi_G * lambda_[None, :]

    rho_sum = np.sum(rho, axis=0)

    if not np.allclose(
        rho_sum,
        1.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        worst = int(np.argmax(np.abs(rho_sum - 1.0)))
        raise RuntimeError(
            "rho normalization failed: "
            f"dimension={worst}, sum={rho_sum[worst]}"
        )

    center_weight = build_center_weights(
        mode=config.center_weight_mode,
        occurrence_weight_mass=occurrence_weight_mass,
        valid_occurrence_count=valid_occurrence_count,
        external_path=config.center_weight_npz,
    )

    center_denominator = np.sum(
        center_weight,
        axis=0,
    )

    mu = (
        np.sum(
            center_weight * G,
            axis=0,
        )
        / center_denominator
    )

    centered = G - mu[None, :]

    variance = (
        np.sum(
            center_weight * np.square(centered),
            axis=0,
        )
        / center_denominator
    )

    variance = np.maximum(
        variance,
        0.0,
    )

    sigma = np.sqrt(variance)

    E = (
        centered
        / (
            sigma[None, :]
            + config.epsilon
        )
    )

    ensure_finite("H", H)
    ensure_finite("lambda", lambda_)
    ensure_finite("rho", rho)
    ensure_finite("mu", mu)
    ensure_finite("sigma", sigma)
    ensure_finite("E", E)

    output_directory = Path(output_directory).resolve()

    output_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f"{output_directory.name}.tmp.",
            dir=str(output_directory.parent),
        )
    )

    try:
        geometry_path = temporary_directory / "geometry.npz"

        np.savez_compressed(
            geometry_path,
            ascii_code=registry.ascii_code,
            dimension_name=registry.dimension_name,
            participation=registry.participation,
            G=G,
            H=H,
            rho=rho,
            mu=mu,
            sigma=sigma,
            E=E,
            center_weight=center_weight,
            occurrence_weight_mass=occurrence_weight_mass,
            valid_occurrence_count=valid_occurrence_count,
            epsilon=np.asarray(
                config.epsilon,
                dtype=np.float64,
            ),
            **{
                "lambda": lambda_,
            },
        )

        compile_manifest = {
            "format": "relational-compiler-v1",
            "ascii_count": ASCII_COUNT,
            "dimension_count": DIMENSION_COUNT,
            "registry": {
                "path": str(registry.source_path),
                "sha256": registry.source_sha256,
            },
            "evidence": [
                {
                    "path": str(shard.path),
                    "sha256": shard.sha256,
                    "occurrences": shard.occurrence_count,
                }
                for shard in shards
            ],
            "configuration": {
                "phi_mode": config.phi_mode,
                "center_weight_mode": config.center_weight_mode,
                "center_weight_npz": (
                    None
                    if config.center_weight_npz is None
                    else str(Path(config.center_weight_npz).resolve())
                ),
                "epsilon": config.epsilon,
            },
            "geometry_shapes": {
                "G": list(G.shape),
                "H": list(H.shape),
                "lambda": list(lambda_.shape),
                "rho": list(rho.shape),
                "mu": list(mu.shape),
                "sigma": list(sigma.shape),
                "E": list(E.shape),
            },
        }

        json_dump(
            temporary_directory / "compile_manifest.json",
            compile_manifest,
        )

        geometry_sha256 = sha256_file(geometry_path)

        frozen_manifest = {
            "format": "relational-frozen-geometry-v1",
            "geometry_file": "geometry.npz",
            "geometry_sha256": geometry_sha256,
            "frozen": True,
        }

        json_dump(
            temporary_directory / "frozen_manifest.json",
            frozen_manifest,
        )

        frozen = FrozenGeometry(temporary_directory)
        frozen.validate()

        if output_directory.exists():
            shutil.rmtree(output_directory)

        temporary_directory.rename(output_directory)

    except Exception:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )
        raise

    return output_directory


class FrozenGeometry:
    def __init__(
        self,
        directory: str | Path,
    ) -> None:

        self.directory = Path(directory).resolve()

        manifest_path = self.directory / "frozen_manifest.json"
        geometry_path = self.directory / "geometry.npz"

        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        if not geometry_path.exists():
            raise FileNotFoundError(geometry_path)

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if not manifest.get("frozen", False):
            raise ValueError("geometry is not marked frozen")

        expected_hash = manifest["geometry_sha256"]
        actual_hash = sha256_file(geometry_path)

        if actual_hash != expected_hash:
            raise ValueError(
                "frozen geometry hash mismatch: "
                "geometry.npz changed after freeze"
            )

        with np.load(geometry_path, allow_pickle=False) as data:
            required = {
                "ascii_code",
                "dimension_name",
                "participation",
                "G",
                "H",
                "lambda",
                "rho",
                "mu",
                "sigma",
                "E",
                "center_weight",
                "occurrence_weight_mass",
                "valid_occurrence_count",
                "epsilon",
            }

            missing = required - set(data.files)

            if missing:
                raise ValueError(
                    "frozen geometry missing arrays: "
                    f"{sorted(missing)}"
                )

            self.ascii_code = np.asarray(
                data["ascii_code"],
                dtype=np.int16,
            )

            self.dimension_name = np.asarray(
                data["dimension_name"]
            ).astype(str)

            self.participation = np.asarray(
                data["participation"],
                dtype=bool,
            )

            self.G = np.asarray(
                data["G"],
                dtype=np.float64,
            )

            self.H = np.asarray(
                data["H"],
                dtype=np.float64,
            )

            self.lambda_ = np.asarray(
                data["lambda"],
                dtype=np.float64,
            )

            self.rho = np.asarray(
                data["rho"],
                dtype=np.float64,
            )

            self.mu = np.asarray(
                data["mu"],
                dtype=np.float64,
            )

            self.sigma = np.asarray(
                data["sigma"],
                dtype=np.float64,
            )

            self.E = np.asarray(
                data["E"],
                dtype=np.float64,
            )

            self.center_weight = np.asarray(
                data["center_weight"],
                dtype=np.float64,
            )

            self.occurrence_weight_mass = np.asarray(
                data["occurrence_weight_mass"],
                dtype=np.float64,
            )

            self.valid_occurrence_count = np.asarray(
                data["valid_occurrence_count"],
                dtype=np.int64,
            )

            self.epsilon = float(
                np.asarray(
                    data["epsilon"]
                ).item()
            )

        self.validate()

    def validate(self) -> None:
        ensure_shape(
            "ascii_code",
            self.ascii_code,
            (ASCII_COUNT,),
        )

        ensure_shape(
            "dimension_name",
            self.dimension_name,
            (DIMENSION_COUNT,),
        )

        ensure_shape(
            "participation",
            self.participation,
            (ASCII_COUNT, DIMENSION_COUNT),
        )

        ensure_shape(
            "G",
            self.G,
            (ASCII_COUNT, DIMENSION_COUNT),
        )

        ensure_shape(
            "H",
            self.H,
            (DIMENSION_COUNT,),
        )

        ensure_shape(
            "lambda",
            self.lambda_,
            (DIMENSION_COUNT,),
        )

        ensure_shape(
            "rho",
            self.rho,
            (ASCII_COUNT, DIMENSION_COUNT),
        )

        ensure_shape(
            "mu",
            self.mu,
            (DIMENSION_COUNT,),
        )

        ensure_shape(
            "sigma",
            self.sigma,
            (DIMENSION_COUNT,),
        )

        ensure_shape(
            "E",
            self.E,
            (ASCII_COUNT, DIMENSION_COUNT),
        )

        ensure_shape(
            "center_weight",
            self.center_weight,
            (ASCII_COUNT, DIMENSION_COUNT),
        )

        ensure_shape(
            "occurrence_weight_mass",
            self.occurrence_weight_mass,
            (ASCII_COUNT, DIMENSION_COUNT),
        )

        ensure_shape(
            "valid_occurrence_count",
            self.valid_occurrence_count,
            (ASCII_COUNT, DIMENSION_COUNT),
        )

        if not np.array_equal(
            self.ascii_code,
            ASCII_CODES,
        ):
            raise ValueError(
                "frozen ASCII population is not exactly 32..126"
            )

        for name, arr in (
            ("G", self.G),
            ("H", self.H),
            ("lambda", self.lambda_),
            ("rho", self.rho),
            ("mu", self.mu),
            ("sigma", self.sigma),
            ("E", self.E),
            ("center_weight", self.center_weight),
            (
                "occurrence_weight_mass",
                self.occurrence_weight_mass,
            ),
        ):
            ensure_finite(name, arr)

        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")

        if np.any(self.H <= 0.0):
            raise ValueError("every frozen H[k] must be > 0")

        if np.any(self.lambda_ <= 0.0):
            raise ValueError("every frozen lambda[k] must be > 0")

        if np.any(self.sigma < 0.0):
            raise ValueError("sigma cannot be negative")

        if not np.allclose(
            self.lambda_,
            1.0 / self.H,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "lambda does not equal reciprocal H"
            )

        rho_sum = np.sum(
            self.rho,
            axis=0,
        )

        if not np.allclose(
            rho_sum,
            1.0,
            rtol=1e-11,
            atol=1e-12,
        ):
            raise ValueError(
                "rho does not sum to one across "
                "the 95-character field"
            )

        center_denominator = np.sum(
            self.center_weight,
            axis=0,
        )

        if np.any(center_denominator <= 0.0):
            raise ValueError(
                "frozen center weights contain "
                "zero-weight dimensions"
            )

        mu_recomputed = (
            np.sum(
                self.center_weight * self.G,
                axis=0,
            )
            / center_denominator
        )

        variance_recomputed = (
            np.sum(
                self.center_weight
                * np.square(
                    self.G
                    - mu_recomputed[None, :]
                ),
                axis=0,
            )
            / center_denominator
        )

        sigma_recomputed = np.sqrt(
            np.maximum(
                variance_recomputed,
                0.0,
            )
        )

        E_recomputed = (
            self.G
            - mu_recomputed[None, :]
        ) / (
            sigma_recomputed[None, :]
            + self.epsilon
        )

        if not np.allclose(
            mu_recomputed,
            self.mu,
            rtol=1e-11,
            atol=1e-12,
        ):
            raise ValueError(
                "frozen mu does not recompute"
            )

        if not np.allclose(
            sigma_recomputed,
            self.sigma,
            rtol=1e-11,
            atol=1e-12,
        ):
            raise ValueError(
                "frozen sigma does not recompute"
            )

        if not np.allclose(
            E_recomputed,
            self.E,
            rtol=1e-11,
            atol=1e-12,
        ):
            raise ValueError(
                "frozen E does not recompute"
            )


def validate_frozen(
    directory: str | Path,
) -> dict:

    frozen = FrozenGeometry(directory)

    center_denominator = np.sum(
        frozen.center_weight,
        axis=0,
    )

    mu_re = (
        np.sum(
            frozen.center_weight * frozen.G,
            axis=0,
        )
        / center_denominator
    )

    sigma_re = np.sqrt(
        np.maximum(
            np.sum(
                frozen.center_weight
                * np.square(
                    frozen.G
                    - mu_re[None, :]
                ),
                axis=0,
            )
            / center_denominator,
            0.0,
        )
    )

    E_re = (
        frozen.G
        - mu_re[None, :]
    ) / (
        sigma_re[None, :]
        + frozen.epsilon
    )

    rho_sum = np.sum(
        frozen.rho,
        axis=0,
    )

    result = {
        "passed": True,
        "G_shape": list(frozen.G.shape),
        "E_shape": list(frozen.E.shape),
        "rho_sums_to_one": bool(
            np.allclose(
                rho_sum,
                1.0,
                rtol=1e-11,
                atol=1e-12,
            )
        ),
        "mu_recomputes": bool(
            np.allclose(
                mu_re,
                frozen.mu,
                rtol=1e-11,
                atol=1e-12,
            )
        ),
        "sigma_recomputes": bool(
            np.allclose(
                sigma_re,
                frozen.sigma,
                rtol=1e-11,
                atol=1e-12,
            )
        ),
        "E_recomputes": bool(
            np.allclose(
                E_re,
                frozen.E,
                rtol=1e-11,
                atol=1e-12,
            )
        ),
        "all_E_finite": bool(
            np.all(
                np.isfinite(
                    frozen.E
                )
            )
        ),
        "all_H_positive": bool(
            np.all(
                frozen.H > 0.0
            )
        ),
        "all_lambda_positive": bool(
            np.all(
                frozen.lambda_ > 0.0
            )
        ),
        "zero_spread_dimension_count": int(
            np.sum(
                frozen.sigma == 0.0
            )
        ),
        "max_abs_rho_sum_error": float(
            np.max(
                np.abs(
                    rho_sum - 1.0
                )
            )
        ),
        "max_abs_E_recompute_error": float(
            np.max(
                np.abs(
                    E_re
                    - frozen.E
                )
            )
        ),
    }

    checks = (
        result["rho_sums_to_one"],
        result["mu_recomputes"],
        result["sigma_recomputes"],
        result["E_recomputes"],
        result["all_E_finite"],
        result["all_H_positive"],
        result["all_lambda_positive"],
    )

    result["passed"] = bool(all(checks))

    return result


def audit_coordinate(
    frozen_directory: str | Path,
    ascii_code: int,
    dimension: int,
) -> dict:

    frozen_directory = Path(
        frozen_directory
    ).resolve()

    frozen = FrozenGeometry(
        frozen_directory
    )

    if dimension < 0 or dimension >= DIMENSION_COUNT:
        raise ValueError(
            f"dimension must be 0..{DIMENSION_COUNT - 1}"
        )

    c = character_index(ascii_code)

    compile_manifest_path = (
        frozen_directory
        / "compile_manifest.json"
    )

    compile_manifest = json.loads(
        compile_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    contributions = []
    recomputed_G = 0.0

    for shard_record in compile_manifest["evidence"]:
        path = Path(shard_record["path"])

        if not path.exists():
            raise FileNotFoundError(
                "source evidence required for audit is missing: "
                f"{path}"
            )

        actual_hash = sha256_file(path)
        expected_hash = shard_record["sha256"]

        if actual_hash != expected_hash:
            raise ValueError(
                "source evidence hash mismatch during trace audit: "
                f"{path}"
            )

        shard = EvidenceShard.load(path)

        rows = np.where(
            shard.y == ascii_code
        )[0]

        for row in rows.tolist():
            measurement = float(
                shard.g[row, dimension]
            )

            if not math.isfinite(measurement):
                continue

            weight = float(shard.w[row])
            weighted_measurement = weight * measurement
            recomputed_G += weighted_measurement

            contributions.append(
                {
                    "shard": str(path),
                    "row": int(row),
                    "source_id": str(
                        shard.source_id[row]
                    ),
                    "ascii_code": int(ascii_code),
                    "character": chr(ascii_code),
                    "dimension": int(dimension),
                    "dimension_name": str(
                        frozen.dimension_name[
                            dimension
                        ]
                    ),
                    "g_i_k": measurement,
                    "w_i": weight,
                    "w_i_times_g_i_k": weighted_measurement,
                }
            )

    stored_G = float(
        frozen.G[c, dimension]
    )

    report = {
        "ascii_code": int(ascii_code),
        "character": chr(ascii_code),
        "character_index": int(c),
        "dimension": int(dimension),
        "dimension_name": str(
            frozen.dimension_name[
                dimension
            ]
        ),
        "trace": {
            "E_c_k": float(
                frozen.E[c, dimension]
            ),
            "sigma_k": float(
                frozen.sigma[dimension]
            ),
            "mu_k": float(
                frozen.mu[dimension]
            ),
            "G_c_k": stored_G,
            "H_k": float(
                frozen.H[dimension]
            ),
            "lambda_k": float(
                frozen.lambda_[dimension]
            ),
            "rho_c_k": float(
                frozen.rho[c, dimension]
            ),
            "center_weight_c_k": float(
                frozen.center_weight[
                    c,
                    dimension,
                ]
            ),
            "occurrence_weight_mass_c_k": float(
                frozen.occurrence_weight_mass[
                    c,
                    dimension,
                ]
            ),
            "valid_occurrence_count_c_k": int(
                frozen.valid_occurrence_count[
                    c,
                    dimension,
                ]
            ),
        },
        "recomputed_G_c_k": float(recomputed_G),
        "absolute_recompute_error": float(
            abs(
                stored_G
                - recomputed_G
            )
        ),
        "source_contributions": contributions,
    }

    return report


def relation_profile_to_state(
    profile: np.ndarray,
    frozen: FrozenGeometry,
    mapper: str,
) -> np.ndarray:
    profile = np.asarray(
        profile,
        dtype=np.float64,
    )

    ensure_shape(
        "profile",
        profile,
        (DIMENSION_COUNT,),
    )

    ensure_finite(
        "profile",
        profile,
    )

    if mapper == "field_zscore":
        return (
            profile
            - frozen.mu
        ) / (
            frozen.sigma
            + frozen.epsilon
        )

    raise ValueError(
        f"unknown mapper {mapper!r}; "
        f"expected one of {STATE_MAPPERS}"
    )


def correspondence_scores(
    h: np.ndarray,
    frozen: FrozenGeometry,
    mode: str,
    omega: np.ndarray | None = None,
) -> np.ndarray:

    h = np.asarray(
        h,
        dtype=np.float64,
    )

    ensure_shape(
        "h",
        h,
        (DIMENSION_COUNT,),
    )

    ensure_finite(
        "h",
        h,
    )

    if omega is None:
        omega = np.ones(
            DIMENSION_COUNT,
            dtype=np.float64,
        )

    else:
        omega = np.asarray(
            omega,
            dtype=np.float64,
        )

        ensure_shape(
            "omega",
            omega,
            (DIMENSION_COUNT,),
        )

        ensure_finite(
            "omega",
            omega,
        )

        if np.any(omega < 0.0):
            raise ValueError(
                "omega must be nonnegative"
            )

    E = frozen.E

    if mode == "negative_squared_distance":
        delta = E - h[None, :]

        return -np.sum(
            omega[None, :]
            * np.square(delta),
            axis=1,
        )

    if mode == "dot":
        return np.sum(
            omega[None, :]
            * E
            * h[None, :],
            axis=1,
        )

    if mode == "cosine":
        sqrt_omega = np.sqrt(omega)

        weighted_E = E * sqrt_omega[None, :]
        weighted_h = h * sqrt_omega

        h_norm = float(
            np.linalg.norm(weighted_h)
        )

        E_norm = np.linalg.norm(
            weighted_E,
            axis=1,
        )

        denominator = E_norm * h_norm

        scores = np.full(
            ASCII_COUNT,
            -np.inf,
            dtype=np.float64,
        )

        valid = denominator > 0.0

        scores[valid] = (
            weighted_E[valid]
            @ weighted_h
        ) / denominator[valid]

        return scores

    raise ValueError(
        f"unknown correspondence mode {mode!r}; "
        f"expected one of {CORRESPONDENCE_MODES}"
    )


def decode_character(
    h_final: np.ndarray,
    frozen: FrozenGeometry,
    correspondence: str,
    omega: np.ndarray | None = None,
) -> tuple[str, int, np.ndarray]:

    scores = correspondence_scores(
        h=h_final,
        frozen=frozen,
        mode=correspondence,
        omega=omega,
    )

    winner_index = int(
        np.argmax(scores)
    )

    ascii_code = int(
        frozen.ascii_code[
            winner_index
        ]
    )

    return (
        chr(ascii_code),
        ascii_code,
        scores,
    )


@dataclass(frozen=True)
class DownstreamCompileGate:
    frozen: FrozenGeometry

    def assert_ready_for_qkv(self) -> None:
        self.frozen.validate()

    def assert_ready_for_ffn(
        self,
        attention_known_information_only: bool,
    ) -> None:

        self.frozen.validate()

        if not attention_known_information_only:
            raise ValueError(
                "FFN compilation blocked: attention has not "
                "been established as operating entirely from "
                "known inference information"
            )


def command_compile(
    args: argparse.Namespace,
) -> int:

    config = CompileConfig(
        phi_mode=args.phi,
        center_weight_mode=args.center_weight,
        epsilon=args.epsilon,
        center_weight_npz=args.center_weight_npz,
    )

    output = compile_geometry(
        registry_path=args.registry,
        evidence_paths=args.evidence,
        output_directory=args.out,
        config=config,
    )

    result = validate_frozen(output)

    print(
        json.dumps(
            {
                "output": str(output),
                "validation": result,
            },
            indent=2,
        )
    )

    return 0


def command_validate(
    args: argparse.Namespace,
) -> int:

    result = validate_frozen(
        args.frozen_directory
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    return 0 if result["passed"] else 1


def command_audit(
    args: argparse.Namespace,
) -> int:

    result = audit_coordinate(
        frozen_directory=args.frozen_directory,
        ascii_code=args.ascii,
        dimension=args.dimension,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    return 0


def command_decode_profile(
    args: argparse.Namespace,
) -> int:

    frozen = FrozenGeometry(
        args.frozen_directory
    )

    with np.load(
        args.profile_npz,
        allow_pickle=False,
    ) as data:

        if "profile" not in data.files:
            raise ValueError(
                "profile NPZ must contain array 'profile'"
            )

        profile = np.asarray(
            data["profile"],
            dtype=np.float64,
        )

    h = relation_profile_to_state(
        profile=profile,
        frozen=frozen,
        mapper=args.mapper,
    )

    character, ascii_code, scores = decode_character(
        h_final=h,
        frozen=frozen,
        correspondence=args.correspondence,
    )

    order = np.argsort(scores)[::-1]

    top = []

    for idx in order[:args.top]:
        code = int(
            frozen.ascii_code[idx]
        )

        top.append(
            {
                "character": chr(code),
                "ascii_code": code,
                "score": float(
                    scores[idx]
                ),
            }
        )

    print(
        json.dumps(
            {
                "prediction": {
                    "character": character,
                    "ascii_code": ascii_code,
                },
                "top": top,
            },
            indent=2,
        )
    )

    return 0


def command_self_test(args: argparse.Namespace) -> int:
    raise RuntimeError(
        "PROHIBITED: synthetic evidence generation is disabled. "
        "Use only real, traceable occurrence evidence. "
        "Missing evidence must remain unresolved."
    )

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Compile and operate a frozen ASCII95 x 437 "
            "relational token geometry."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    p = sub.add_parser(
        "compile",
        help=(
            "Compile occurrence evidence into frozen "
            "95x437 relational geometry."
        ),
    )

    p.add_argument(
        "--registry",
        required=True,
    )

    p.add_argument(
        "--evidence",
        nargs="+",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    p.add_argument(
        "--phi",
        choices=PHI_MODES,
        required=True,
    )

    p.add_argument(
        "--center-weight",
        choices=CENTER_WEIGHT_MODES,
        required=True,
    )

    p.add_argument(
        "--center-weight-npz",
        default=None,
    )

    p.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
    )

    p.set_defaults(
        func=command_compile
    )

    p = sub.add_parser(
        "validate-frozen",
        help=(
            "Recompute and validate the frozen geometry "
            "invariants."
        ),
    )

    p.add_argument(
        "frozen_directory",
    )

    p.set_defaults(
        func=command_validate
    )

    p = sub.add_parser(
        "audit-coordinate",
        help=(
            "Trace one E[c,k] coordinate back through "
            "G[c,k] to source occurrence evidence."
        ),
    )

    p.add_argument(
        "frozen_directory",
    )

    p.add_argument(
        "--ascii",
        type=int,
        required=True,
    )

    p.add_argument(
        "--dimension",
        type=int,
        required=True,
    )

    p.set_defaults(
        func=command_audit
    )

    p = sub.add_parser(
        "decode-profile",
        help=(
            "Construct a state from a known-information profile "
            "and compare it against all 95 frozen token positions."
        ),
    )

    p.add_argument(
        "frozen_directory",
    )

    p.add_argument(
        "--profile-npz",
        required=True,
    )

    p.add_argument(
        "--mapper",
        choices=STATE_MAPPERS,
        required=True,
    )

    p.add_argument(
        "--correspondence",
        choices=CORRESPONDENCE_MODES,
        required=True,
    )

    p.add_argument(
        "--top",
        type=int,
        default=10,
    )

    p.set_defaults(
        func=command_decode_profile
    )

    p = sub.add_parser(
        "self-test",
        help="Run compiler integrity tests.",
    )

    p.set_defaults(
        func=command_self_test
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
