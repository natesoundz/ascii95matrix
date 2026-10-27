"""Weight-source backends for the compiled conical operators."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Protocol

import numpy as np

from .runtime import CompiledModel


CHANNEL_NAMES = (
    "channel:E",
    "channel:B:Q",
    "channel:B:K",
    "channel:B:V",
    "channel:B:O",
    "channel:B:F1",
    "channel:B:F2",
)


class WeightSource(Protocol):
    def get(self, name: str) -> np.ndarray:
        """Return a read-only weight array."""

    def describe(self) -> dict:
        """Describe the active weight source."""


class DirectWeights:
    def __init__(self, model: CompiledModel):
        self.model = model

    def get(self, name: str) -> np.ndarray:
        if name not in CHANNEL_NAMES:
            raise KeyError(f"Unsupported channel weight: {name}")
        array = self.model._segment_array(name)
        array.setflags(write=False)
        return array

    def describe(self) -> dict:
        return {
            "mode": "direct",
            "tensors": list(CHANNEL_NAMES),
        }


def _load_rubik_module(path: str | Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("sqlir_user_rubik_store", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Rubik module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown module"
        raise RuntimeError(
            f"Could not import Rubik module {path}: missing module {missing}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Could not import Rubik module {path}: {exc}") from exc
    if not hasattr(module, "RubikStore"):
        raise RuntimeError(f"Rubik module {path} does not define RubikStore")
    return module


class RubikWeights:
    def __init__(self, store_dir: str | Path, rubik_module_path: str | Path):
        self.store_dir = Path(store_dir)
        self.rubik_module_path = Path(rubik_module_path)
        self.module = _load_rubik_module(self.rubik_module_path)
        try:
            self.store = self.module.RubikStore.open(str(self.store_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Could not open Rubik store {self.store_dir}: {exc}"
            ) from exc

    def get(self, name: str) -> np.ndarray:
        try:
            self.store.address(name)
            array = self.store.materialize(name)
        except Exception as exc:
            raise RuntimeError(f"Could not read Rubik tensor {name!r}: {exc}") from exc
        array = np.asarray(array)
        array.setflags(write=False)
        return array

    def describe(self) -> dict:
        return {
            "mode": "rubik",
            "store_dir": str(self.store_dir),
            "format": self.store.manifest.get("format"),
            "tensors": {
                item["name"]: item.get("sha256")
                for item in self.store.manifest.get("tensors", [])
            },
            "journal_event_count": len(self.store.journal),
        }


def export_rubik(
    model_dir: str | Path,
    out_dir: str | Path,
    rubik_module_path: str | Path,
) -> Path:
    module = _load_rubik_module(rubik_module_path)
    model = CompiledModel(model_dir, verify="full")
    names = sorted(CHANNEL_NAMES)
    tensors = {name: np.asarray(model._segment_array(name)) for name in names}
    banks = {name: index for index, name in enumerate(names)}
    module.RubikStore.build(tensors, str(out_dir), banks=banks)
    return Path(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--model", required=True, type=Path)
    export.add_argument("--out", required=True, type=Path)
    export.add_argument("--rubik-module", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "export":
        export_rubik(args.model, args.out, args.rubik_module)


if __name__ == "__main__":
    main()
