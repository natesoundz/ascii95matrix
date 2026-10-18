"""
Workbook-as-L0 for HSCM — the Canonical ASCII95 workbook as the base layer.

Replaces the role L0_PRIMITIVES played in matrix.py (65 semantic atoms):
here L0 is the full 437-dimension typed attribute registry. Everything
above this layer references characters and dimensions BY POINTER
(canonical dimension id / character), never by copied value, in the
spirit of config.py's DictPointer discipline.

Constraint semantics come from the registry itself:
  X  = structurally impossible -> binding is FORBIDDEN (hard gate)
  1/0 = fixed boolean          -> resolved by definition
  K  = identity key            -> never a weight input
  E/M/R/U/V/S                  -> bindable only from warranted evidence,
                                  each with its declared resolver type

The layer loads from the registry export (registry_dimensions.tsv +
binding_manifest.json, which mirror the workbook cell-for-cell; the
loader verifies counts and id sets against the TSV).
"""

import csv
import json
from pathlib import Path

REGISTRY_DIR = Path(__file__).parent / ".." / ".." / "workbook_audit" / "repo_copy" / "data"

RESOLVED_CODES = {"1", "0", "K"}
EVIDENCE_CODES = {"E", "M", "R", "U", "V", "S"}


class Attribute:
    """One workbook dimension (a column of the registry)."""
    __slots__ = ("index", "id", "name", "interaction", "eligibility",
                 "defer_scope", "status_by_ascii", "section")

    def __init__(self, row, status_by_ascii, section):
        self.index = int(row["index"])
        self.id = row["id"]
        self.name = row["name"]
        self.interaction = row["interaction"]
        self.eligibility = ""
        self.defer_scope = row["defer_scope"]
        self.status_by_ascii = status_by_ascii
        self.section = section

    def __repr__(self):
        return f"Attribute({self.index}, {self.id!r})"


class WorkbookL0:
    """The workbook as HSCM's L0: 95 characters x 437 typed dimensions."""

    def __init__(self, registry_dir=REGISTRY_DIR):
        registry_dir = Path(registry_dir)
        tsv_rows = list(csv.DictReader(
            open(registry_dir / "registry_dimensions.tsv"), delimiter="\t"))
        manifest = json.load(open(registry_dir / "binding_manifest.json"))
        man_dims = {d["id"]: d for d in manifest["dimensions"]}
        assert manifest["dimension_count"] == 437 == len(tsv_rows), "registry size drift"

        # Sections from the workbook's group headers (sheet columns 48-150 etc.
        # are recorded in the audit); section = namespace path's middle part.
        self.attributes = {}          # dim_id -> Attribute   (the pointer table)
        self.by_index = []            # index  -> Attribute
        for row in tsv_rows:
            d = man_dims[row["id"]]
            assert d["name"] == row["name"], f"name drift on {row['id']}"
            section = row["id"].split(".")[1] if "." in row["id"] else ""
            attr = Attribute(row, d["status_by_ascii"], section)
            attr.eligibility = d.get("eligibility", "")
            self.attributes[attr.id] = attr
            self.by_index.append(attr)
        assert [a.index for a in self.by_index] == list(range(437))

        # The 95 characters, in workbook row order (ASCII 32..126).
        self.chars = [chr(c) for c in range(32, 127)]
        self._char_row = {ch: i for i, ch in enumerate(self.chars)}

    # -- constraint API -------------------------------------------------
    def status(self, char, dim_id):
        """The registry's own status code for (char, dimension)."""
        return self.attributes[dim_id].status_by_ascii[self._char_row[char]]

    def bindable(self, char, dim_id):
        """Hard gate: X means structurally impossible — never bindable."""
        return self.status(char, dim_id) != "X"

    def resolved(self, char, dim_id):
        return self.status(char, dim_id) in RESOLVED_CODES

    def requires_evidence(self, char, dim_id):
        return self.status(char, dim_id) in EVIDENCE_CODES

    # -- sparse pointer views -------------------------------------------
    def pointers(self, char):
        """Sparse L0 address of a character: only its non-X dimensions,
        as {dim_id: status}. X entries are absent (impossible), exactly
        like HSCM's sparse adjacency rule: everything not listed is zero."""
        r = self._char_row[char]
        return {a.id: a.status_by_ascii[r] for a in self.by_index
                if a.status_by_ascii[r] != "X"}

    def dimensions(self, namespace=None):
        if namespace is None:
            return list(self.by_index)
        return [a for a in self.by_index if a.id.startswith(namespace + ".")]

    def phonetic_block(self):
        """The 103 physical features: registry indices 43..145."""
        return self.by_index[43:146]

    def summary(self):
        from collections import Counter
        codes = Counter()
        for a in self.by_index:
            codes.update(a.status_by_ascii)
        return {"dimensions": len(self.by_index), "characters": len(self.chars),
                "status_cells": dict(codes),
                "phonetic_block": len(self.phonetic_block())}


if __name__ == "__main__":
    l0 = WorkbookL0()
    print(json.dumps(l0.summary(), indent=1))
    print("pointers('a'):", len(l0.pointers("a")), "non-X dimensions")
    print("pointers('~'):", len(l0.pointers("~")), "non-X dimensions")
    print("bindable('5', phonetic bilabial):",
          l0.bindable("5", "phonetic.consonant_place_of_articulation.bilabial"),
          "| status:", l0.status("5", "phonetic.consonant_place_of_articulation.bilabial"))
