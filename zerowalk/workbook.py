"""Canonical 95-by-437 workbook feature encoding."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STATUS_CELLS = {"E", "K", "M", "R", "S", "U", "V", "X"}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "data" / "registry_dimensions.tsv"


@dataclass(frozen=True)
class WorkbookEncoding:
    characters: tuple[str, ...]
    feature_names: tuple[str, ...]
    features: np.ndarray
    profile_id: np.ndarray
    profile_table: np.ndarray
    groups: tuple[tuple[str, ...], ...]
    boolean_features: int
    one_hot_features: int

    @property
    def audit_text(self) -> str:
        groups = "\n".join("  " + "".join(group) for group in self.groups)
        return (
            f"features: {self.features.shape[1]} "
            f"({self.boolean_features} bool + {self.one_hot_features} one-hot)\n"
            f"distinct row profiles: {len(self.groups)}; rank: "
            f"{int(np.linalg.matrix_rank(self.features))}\n"
            "These groups are indistinguishable from defined evidence. "
            "Separating them requires binding E/R/M/V cells. Do not invent "
            "those values.\n"
            f"groups:\n{groups}"
        )


def load_workbook(
    workbook_path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> WorkbookEncoding:
    workbook_path = Path(workbook_path)
    registry_path = Path(registry_path)
    raw = json.loads(workbook_path.read_text(encoding="utf-8"))
    attribute_ids = raw["attribute_ids"]
    records = sorted(raw["records"], key=lambda record: record["ascii"])
    if len(attribute_ids) != 437 or len(records) != 95:
        raise ValueError("expected a 95-row, 437-column canonical workbook")
    if [record["ascii"] for record in records] != list(range(32, 127)):
        raise ValueError("workbook records must be in ASCII order 32..126")

    with registry_path.open(encoding="utf-8", newline="") as stream:
        dimensions = list(csv.DictReader(stream, delimiter="\t"))
    registry_ids = [dimension["id"] for dimension in dimensions]
    if registry_ids != attribute_ids:
        raise ValueError("workbook attribute ids and registry ids differ in order")
    if len(dimensions) != 437:
        raise ValueError("registry must contain 437 dimensions")
    key_columns = [
        index for index, dimension in enumerate(dimensions)
        if dimension["interaction"] == "KEY"
    ]
    if key_columns != [0, 1, 2, 3]:
        raise ValueError(f"expected identity key columns 0..3, found {key_columns}")

    cells = [record["cells"] for record in records]
    if any(len(row) != 437 for row in cells):
        raise ValueError("each workbook row must contain 437 cells")
    feature_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    boolean_count = 0
    one_hot_count = 0

    for column in range(4, 437):
        column_id = attribute_ids[column]
        values = [row[column] for row in cells]
        if any(
            isinstance(value, str) and value in {"0", "1"}
            for value in values
        ):
            encoded = np.asarray(
                [1 if isinstance(value, str) and value == "1"
                 else -1 if isinstance(value, str) and value == "0"
                 else 0 for value in values],
                dtype=np.int64,
            )
            if np.any(encoded):
                feature_columns.append(encoded)
                feature_names.append(f"{column_id}=bool")
                boolean_count += 1

        supplied_values = sorted({
            str(value) for value in values
            if not (isinstance(value, str) and
                    (value in STATUS_CELLS or value in {"0", "1"}))
        })
        for value in supplied_values:
            encoded = np.asarray(
                [1 if str(cell) == value and not (
                    isinstance(cell, str) and
                    (cell in STATUS_CELLS or cell in {"0", "1"})
                ) else 0 for cell in values],
                dtype=np.int64,
            )
            feature_columns.append(encoded)
            feature_names.append(f"{column_id}={value}")
            one_hot_count += 1

    features = np.column_stack(feature_columns)
    profile_lookup: dict[bytes, int] = {}
    profile_rows: list[np.ndarray] = []
    profile_id = np.empty(95, dtype=np.int64)
    for row_index, feature_row in enumerate(features):
        key = feature_row.tobytes()
        if key not in profile_lookup:
            profile_lookup[key] = len(profile_rows)
            profile_rows.append(feature_row.copy())
        profile_id[row_index] = profile_lookup[key]
    profile_table = np.vstack(profile_rows)
    groups_list: list[list[str]] = [[] for _ in profile_rows]
    for row_index, record in enumerate(records):
        groups_list[int(profile_id[row_index])].append(record["character"])

    if features.shape != (95, 94) or boolean_count != 21 or one_hot_count != 73:
        raise ValueError(
            f"unexpected encoding shape/counts: {features.shape}, "
            f"{boolean_count} bool, {one_hot_count} one-hot"
        )
    if profile_table.shape != (26, 94) or np.linalg.matrix_rank(features) != 26:
        raise ValueError("unexpected defined-evidence profile count or rank")

    return WorkbookEncoding(
        characters=tuple(record["character"] for record in records),
        feature_names=tuple(feature_names),
        features=features,
        profile_id=profile_id,
        profile_table=profile_table,
        groups=tuple(tuple(group) for group in groups_list),
        boolean_features=boolean_count,
        one_hot_features=one_hot_count,
    )
