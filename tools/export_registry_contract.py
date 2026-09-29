#!/usr/bin/env python3
"""Export the canonical ASCII95 workbook into a machine-readable registry contract.

The workbook remains authoritative. This script does not assign embedding
coordinates and does not reinterpret deferred/relational cells as numeric data.
It only exports the workbook's declared metadata and per-character status cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ASCII_MIN = 32
ASCII_MAX = 126
EXPECTED_DIMENSIONS = 437
VALID_STATUS = set("X01ERMUVKS")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def colnum(label: str) -> int:
    n = 0
    for ch in label:
        n = n * 26 + ord(ch.upper()) - 64
    return n


def load_grid(path: Path, sheet_name: str) -> dict[str, str]:
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    relns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as z:
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        sheet = next(
            s for s in workbook.find("s:sheets", ns)
            if s.get("name") == sheet_name
        )
        target = targets[sheet.attrib[f"{{{relns}}}id"]]
        sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target

        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = [
                "".join(si.itertext())
                for si in ET.fromstring(z.read("xl/sharedStrings.xml"))
            ]

        grid: dict[str, str] = {}
        root = ET.fromstring(z.read(sheet_path))
        for row in root.findall("s:sheetData/s:row", ns):
            for cell in row:
                if cell.find("s:f", ns) is not None:
                    raise ValueError(
                        "canonical registry contains a formula; export requires resolved source cells"
                    )
                value = cell.find("s:v", ns)
                inline = cell.find("s:is", ns)
                if cell.get("t") == "s" and value is not None:
                    val = shared[int(value.text)]
                elif inline is not None:
                    val = "".join(inline.itertext())
                else:
                    val = value.text if value is not None else None
                if val is not None:
                    grid[cell.attrib["r"]] = str(val)
    return grid


def export_contract(workbook: Path, output: Path) -> dict:
    grid = load_grid(workbook, "ASCII95 Matrix")

    columns = sorted(
        (
            cell[:-1]
            for cell in grid
            if cell.endswith("8")
            and cell[:-1].isalpha()
            and colnum(cell[:-1]) >= 6
        ),
        key=colnum,
    )
    ids = [grid[c + "8"] for c in columns]
    if len(ids) != EXPECTED_DIMENSIONS:
        raise ValueError(f"expected {EXPECTED_DIMENSIONS} dimensions, got {len(ids)}")
    if len(set(ids)) != EXPECTED_DIMENSIONS:
        raise ValueError("canonical dimension IDs are not unique")

    codes = [int(grid[f"C{r}"]) for r in range(10, 105)]
    if codes != list(range(ASCII_MIN, ASCII_MAX + 1)):
        raise ValueError("ASCII rows are not exactly 32..126 in ascending order")

    characters = []
    for row, code in zip(range(10, 105), codes):
        characters.append(
            {
                "ascii_code": code,
                "character": chr(code),
                "row": row,
                "name": grid.get(f"D{row}", ""),
            }
        )

    dimensions = []
    global_counts = Counter()
    for index, col in enumerate(columns):
        status = [grid.get(f"{col}{row}", "") for row in range(10, 105)]
        unknown = sorted(set(status) - VALID_STATUS)
        if unknown:
            raise ValueError(f"{ids[index]} has unknown status cells: {unknown}")

        counts = Counter(status)
        global_counts.update(status)
        dimensions.append(
            {
                "index": index,
                "column": col,
                "id": ids[index],
                "name": grid.get(col + "7", ""),
                "interaction": grid.get(col + "3", ""),
                "eligibility": grid.get(col + "4", ""),
                "metadata_row5": grid.get(col + "5", ""),
                "defer_scope": grid.get(col + "6", ""),
                "status_counts": dict(sorted(counts.items())),
                "status_by_ascii": status,
                "eligible_ascii": [
                    code for code, s in zip(codes, status) if s != "X"
                ],
                "fixed_positive_ascii": [
                    code for code, s in zip(codes, status) if s == "1"
                ],
                "fixed_negative_ascii": [
                    code for code, s in zip(codes, status) if s == "0"
                ],
                "deferred_ascii": [
                    code for code, s in zip(codes, status) if s == "E"
                ],
                "relation_ascii": [
                    code for code, s in zip(codes, status) if s == "R"
                ],
                "measurement_ascii": [
                    code for code, s in zip(codes, status) if s == "M"
                ],
                "modulator_ascii": [
                    code for code, s in zip(codes, status) if s == "U"
                ],
                "value_slot_ascii": [
                    code for code, s in zip(codes, status) if s == "V"
                ],
                "identity_key_ascii": [
                    code for code, s in zip(codes, status) if s == "K"
                ],
                "structural_ascii": [
                    code for code, s in zip(codes, status) if s == "S"
                ],
            }
        )

    by_interaction = Counter(d["interaction"] for d in dimensions)
    by_defer_scope = Counter(d["defer_scope"] for d in dimensions)

    contract = {
        "format": "ascii95-canonical-registry-contract-v1",
        "source_workbook": str(workbook.as_posix()),
        "source_sha256": sha256_file(workbook),
        "ascii_min": ASCII_MIN,
        "ascii_max": ASCII_MAX,
        "vocabulary_size": len(characters),
        "dimension_count": len(dimensions),
        "allowed_status_cells": sorted(VALID_STATUS),
        "interpretation": {
            "registry_is_embedding_coordinates": False,
            "status_cells_are_numeric_coordinates": False,
            "X": "structurally ineligible",
            "1": "fixed positive declared evidence",
            "0": "fixed negative declared evidence",
            "E": "eligible deferred; bind occurrence/context source before numeric measurement",
            "R": "typed relation; retain participants/relation until projection is declared",
            "M": "measurement; bind realization axis, units and deterministic transform",
            "U": "modulator/statistic; not an independent primitive coordinate",
            "V": "scalar/category slot; bind actual value and declared encoding",
            "K": "identity key; identifier, not a weight input",
            "S": "structural/derived; resolve from declared dependencies",
        },
        "global_status_counts": dict(sorted(global_counts.items())),
        "interaction_counts": dict(sorted(by_interaction.items())),
        "defer_scope_counts": dict(sorted(by_defer_scope.items())),
        "characters": characters,
        "dimensions": dimensions,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return contract


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--workbook",
        default="data/Canonical_ASCII95_Attribute_Matrix_v1.xlsx",
        type=Path,
    )
    p.add_argument(
        "--out",
        default="data/registry_contract.json",
        type=Path,
    )
    args = p.parse_args()
    contract = export_contract(args.workbook, args.out)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "sha256": contract["source_sha256"],
                "characters": contract["vocabulary_size"],
                "dimensions": contract["dimension_count"],
                "global_status_counts": contract["global_status_counts"],
                "interaction_counts": contract["interaction_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
