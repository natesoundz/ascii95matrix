"""Memory-mapped runtime for compiled SQL-IR artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import canonical_key
from .conical_sql_ir_transformer import norm, relu, softmax_causal


DISCLOSURES = [
    "This is a deterministic SQL-derived projection prototype, not an equivalent trained transformer.",
    "Typed workbook unresolved cells are retained in source JSON and numerically inactive; known categorical cells activate their own attribute column.",
    "SQL rows map through character histograms and explicit workbook column activation: C_ij=sum_ab T_ab E_ai E_bj; B=row_normalize(C).",
    "Character-histogram correspondence is a deterministic engineering choice, not proof of linguistic semantics.",
    "Nearest workbook character is not a validated language decoder.",
    "Full database results used unless --limit is explicitly supplied.",
]


class CompiledModel:
    def __init__(self, dir: str | Path, verify: str = "lazy"):
        if verify not in {"lazy", "full", "none"}:
            raise ValueError("verify must be 'lazy', 'full', or 'none'")
        self.directory = Path(dir)
        self.manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("format_version") != 2:
            raise ValueError(
                f"Unsupported model format version: {self.manifest.get('format_version')}"
            )
        self._verify_policy = verify
        self._blob = np.memmap(
            self.directory / "blob.bin", dtype=np.uint8, mode="r"
        )
        self._verified: set[str] = set()
        self._queries = {query["id"]: query for query in self.manifest["queries"]}
        if verify == "full":
            self.verify_all()

    def _verify_segment(self, name: str, force: bool = False) -> None:
        if self._verify_policy == "none" and not force:
            return
        if name in self._verified:
            return
        descriptor = self.manifest["segments"][name]
        start = descriptor["offset"]
        end = start + descriptor["nbytes"]
        actual = hashlib.sha256(memoryview(self._blob)[start:end]).hexdigest()
        if actual != descriptor["sha256"]:
            raise ValueError(f"Segment SHA-256 mismatch: {name}")
        self._verified.add(name)

    def verify_all(self) -> None:
        for name in self.manifest["segments"]:
            self._verify_segment(name, force=True)

    def _segment_bytes(self, name: str) -> memoryview:
        self._verify_segment(name)
        descriptor = self.manifest["segments"][name]
        start = descriptor["offset"]
        end = start + descriptor["nbytes"]
        return memoryview(self._blob)[start:end]

    def _segment_array(self, name: str) -> np.ndarray:
        descriptor = self.manifest["segments"][name]
        raw = self._segment_bytes(name)
        return np.frombuffer(raw, dtype=np.dtype(descriptor["dtype"])).reshape(
            descriptor["shape"]
        )

    def list_queries(self) -> list[dict]:
        return self.manifest["queries"]

    def _query_keys(self, query: dict) -> list[Any]:
        keys_name, offsets_name = query["segments"][:2]
        keys = self._segment_bytes(keys_name)
        offsets = self._segment_array(offsets_name)
        return [
            json.loads(bytes(keys[offsets[i] : offsets[i + 1]]).decode("utf-8"))
            for i in range(query["n_keys"])
        ]

    def _find_key(self, query: dict, key: Any) -> tuple[int, int] | None:
        if query["kind"] == "text":
            key = str(key).lower()
        elif len(query["key_columns"]) > 1:
            key = list(key) if isinstance(key, (tuple, list)) else key
        encoded = canonical_key(key)
        keys_name, offsets_name, indptr_name, _ = query["segments"]
        keys = self._segment_bytes(keys_name)
        offsets = self._segment_array(offsets_name)
        indptr = self._segment_array(indptr_name)
        low, high = 0, query["n_keys"]
        while low < high:
            middle = (low + high) // 2
            candidate = bytes(keys[offsets[middle] : offsets[middle + 1]])
            if candidate < encoded:
                low = middle + 1
            else:
                high = middle
        if low >= query["n_keys"]:
            return None
        if bytes(keys[offsets[low] : offsets[low + 1]]) != encoded:
            return None
        return int(indptr[low]), int(indptr[low + 1])

    def _rows_by_rowids(self, table: str, rowids: np.ndarray) -> list:
        schema = self.manifest["tables"][table]
        segments = schema["segments"]
        stored_rowids = self._segment_array(segments["rowids"])
        offsets = self._segment_array(segments["offsets"])
        rows = self._segment_bytes(segments["rows"])
        result = []
        for rowid in rowids:
            position = int(np.searchsorted(stored_rowids, rowid))
            if position >= len(stored_rowids) or int(stored_rowids[position]) != int(rowid):
                raise KeyError(f"Unknown row id {rowid} for table {table}")
            start = int(offsets[position])
            end = (
                int(offsets[position + 1])
                if position + 1 < len(offsets)
                else len(rows)
            )
            result.append(json.loads(bytes(rows[start:end]).decode("utf-8")))
        return result

    @staticmethod
    def _validate_pagination(offset: int, limit: int | None) -> None:
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
        ):
            raise ValueError("limit must be a nonnegative integer or None")

    def _result_rows(self, query: dict, start: int, end: int) -> list:
        rows_name, offsets_name = query["segments"]
        rows = self._segment_bytes(rows_name)
        offsets = self._segment_array(offsets_name)
        return [
            json.loads(
                bytes(rows[int(offsets[index]) : int(offsets[index + 1])]).decode(
                    "utf-8"
                )
            )
            for index in range(start, end)
        ]

    def lookup(
        self,
        query_id: str,
        key: Any = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list:
        self._validate_pagination(offset, limit)
        if query_id not in self._queries:
            raise KeyError(f"Unknown query id: {query_id}")
        query = self._queries[query_id]
        if query["kind"] in {"fixed", "count"}:
            offsets = self._segment_array(query["segments"][1])
            count = len(offsets) - 1
            start = min(offset, count)
            end = count if limit is None else min(count, start + limit)
            return self._result_rows(query, start, end)
        bounds = self._find_key(query, key)
        if bounds is None:
            return []
        _, _, _, indices_name = query["segments"]
        indices = self._segment_array(indices_name)
        start = bounds[0] + offset
        end = bounds[1] if limit is None else min(bounds[1], start + limit)
        if start >= end:
            return []
        return self._rows_by_rowids(query["table"], indices[start:end])

    def lookup_count(self, query_id: str, key: Any = None) -> int:
        if query_id not in self._queries:
            raise KeyError(f"Unknown query id: {query_id}")
        query = self._queries[query_id]
        if query["kind"] in {"fixed", "count"}:
            return len(self._segment_array(query["segments"][1])) - 1
        bounds = self._find_key(query, key)
        return 0 if bounds is None else bounds[1] - bounds[0]

    def row(self, table: str, rowid: Any) -> list:
        if table not in self.manifest["tables"]:
            raise KeyError(f"Unknown table: {table}")
        schema = self.manifest["tables"][table]
        if schema["without_rowid"]:
            rows = self.lookup(f"pk:{table}", rowid)
            if not rows:
                raise KeyError(f"Unknown primary key {rowid!r} for table {table}")
            return rows[0]
        segments = schema["segments"]
        stored_rowids = self._segment_array(segments["rowids"])
        position = int(np.searchsorted(stored_rowids, rowid))
        if position >= len(stored_rowids) or int(stored_rowids[position]) != rowid:
            raise KeyError(f"Unknown row id {rowid} for table {table}")
        offsets = self._segment_array(segments["offsets"])
        raw = self._segment_bytes(segments["rows"])
        start = int(offsets[position])
        end = int(offsets[position + 1]) if position + 1 < len(offsets) else len(raw)
        return json.loads(bytes(raw[start:end]).decode("utf-8"))

    def forward(self, text: str) -> dict:
        if any(ord(character) < 32 or ord(character) > 126 for character in text):
            raise ValueError("Only ASCII 32..126 is supported by the workbook.")
        if not text:
            raise ValueError("Nonempty input required")

        E = self._segment_array("channel:E")
        X = E[np.array([ord(character) - 32 for character in text])]
        Z1 = norm(X)

        def project(value: np.ndarray, name: str) -> np.ndarray:
            return value @ self._segment_array(f"channel:B:{name}")

        Q = project(Z1, "Q")
        K = project(Z1, "K")
        V = project(Z1, "V")
        A = softmax_causal((Q @ K.T) / np.sqrt(437))
        H = A @ V
        O = project(H, "O")
        Z2 = X + O
        Z3 = norm(Z2)
        U = relu(project(Z3, "F1"))
        delta = project(U, "F2")
        Y = Z2 + delta
        if Y.shape != X.shape or not np.isfinite(Y).all():
            raise ValueError("Forward pass produced invalid output")
        distance = ((Y[:, None, :] - E[None, :, :]) ** 2).sum(axis=2)
        nearest = "".join(
            chr(32 + int(index)) for index in distance.argmin(axis=1)
        )
        operators = {
            name: self._segment_array(f"channel:B:{name}")
            for name in self.manifest["compiled_rows"]
        }
        return {
            "status": "EXECUTED",
            "input": text,
            "shape": list(Y.shape),
            "compiled_rows": self.manifest["compiled_rows"],
            "operator_sha256": {
                name: hashlib.sha256(operator.tobytes()).hexdigest()
                for name, operator in operators.items()
            },
            "operator_invariants": {
                name: {
                    "min": float(operator.min()),
                    "max_row_sum_error": float(
                        np.max(np.abs(operator.sum(axis=1) - 1))
                    ),
                }
                for name, operator in operators.items()
            },
            "input_vectors": X.tolist(),
            "output_vectors": Y.tolist(),
            "attention": A.tolist(),
            "nearest_workbook_char_diagnostic": nearest,
            "disclosures": DISCLOSURES,
        }
