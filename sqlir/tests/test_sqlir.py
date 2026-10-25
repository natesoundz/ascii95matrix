"""TEST-ONLY fixture: validates software only and must never be compiled into a model artifact."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sqlir.compile import compile_model
from sqlir.runtime import CompiledModel
from sqlir.verify import verify_model


ROOT = Path(__file__).resolve().parents[2]
SQLIR = ROOT / "sqlir"
WORKBOOK = SQLIR / "conical437_real_workbook.json"
REFERENCE_SCRIPT = SQLIR / "conical_sql_ir_transformer.py"


def create_fixture(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE lexeme (
          lexeme_id INTEGER PRIMARY KEY,
          normalized TEXT
        );
        CREATE TABLE collocation (
          collocation_id INTEGER PRIMARY KEY,
          left_lemma TEXT,
          right_lemma TEXT,
          frequency REAL
        );
        CREATE TABLE relation (
          relation_id INTEGER PRIMARY KEY,
          source_lexeme_id INTEGER,
          target_lemma TEXT
        );
        CREATE TABLE webster_entry (
          webster_id INTEGER PRIMARY KEY,
          normalized TEXT
        );
        CREATE TABLE definition_term (
          webster_id INTEGER,
          term_webster_id INTEGER,
          term_count INTEGER,
          PRIMARY KEY (webster_id, term_webster_id)
        );
        CREATE TABLE occurrence (
          occurrence_id INTEGER PRIMARY KEY,
          lemma TEXT,
          example_id INTEGER,
          token_index INTEGER,
          head_index INTEGER
        );
        CREATE TABLE inflection_form (
          inflection_id INTEGER PRIMARY KEY,
          lexeme_id INTEGER,
          normalized TEXT
        );
        CREATE TABLE indexed_parent (
          parent_id INTEGER PRIMARY KEY,
          label TEXT
        );
        CREATE TABLE indexed_child (
          child_id INTEGER PRIMARY KEY,
          parent_id INTEGER REFERENCES indexed_parent(parent_id),
          value TEXT,
          payload BLOB
        );
        CREATE INDEX indexed_child_parent_idx ON indexed_child(parent_id);
        CREATE TABLE without_rowid_store (
          code TEXT PRIMARY KEY,
          note TEXT
        ) WITHOUT ROWID;
        """
    )
    words = [
        "Earth", "sun", "moon", "world", "hello", "language", "book", "word",
        "form", "surface", "headword", "lemma", "house", "river", "mountain",
        "tree", "water", "fire", "wind", "stone", "light", "dark", "time",
        "space", "sound", "music", "letter", "number", "green", "blue",
        "café", "naïve", "façade", "résumé", "雪", "unicode",
    ]
    con.executemany(
        "INSERT INTO lexeme(lexeme_id, normalized) VALUES (?, ?)",
        [(i + 1, word) for i, word in enumerate(words)],
    )
    con.executemany(
        "INSERT INTO collocation VALUES (?, ?, ?, ?)",
        [
            (i + 1, words[i], words[(i + 1) % len(words)], float(i + 1))
            for i in range(24)
        ],
    )
    con.executemany(
        "INSERT INTO relation VALUES (?, ?, ?)",
        [
            (i + 1, i + 1, None if i == 2 else words[(i + 2) % len(words)])
            for i in range(18)
        ],
    )
    con.executemany(
        "INSERT INTO webster_entry VALUES (?, ?)",
        [(i + 1, words[i]) for i in range(20)],
    )
    con.executemany(
        "INSERT INTO definition_term VALUES (?, ?, ?)",
        [(i + 1, i + 2, i % 4 + 1) for i in range(1, 16)],
    )
    con.executemany(
        "INSERT INTO occurrence VALUES (?, ?, ?, ?, ?)",
        [
            (
                i + 1,
                words[i % len(words)],
                i // 3,
                i % 3,
                None if i % 5 == 0 else (i % 3),
            )
            for i in range(30)
        ],
    )
    con.executemany(
        "INSERT INTO inflection_form VALUES (?, ?, ?)",
        [(i + 1, i + 1, words[i] + ("s" if i % 2 else "ed")) for i in range(20)],
    )
    con.executemany(
        "INSERT INTO indexed_parent VALUES (?, ?)",
        [(i, None if i == 3 else f"parent-{i}") for i in range(1, 9)],
    )
    con.executemany(
        "INSERT INTO indexed_child VALUES (?, ?, ?, ?)",
        [
            (
                i,
                None if i == 4 else (i % 8) + 1,
                "unicode-雪" if i == 2 else f"value-{i}",
                None if i == 5 else bytes([i, 0, 255]),
            )
            for i in range(1, 21)
        ],
    )
    con.executemany(
        "INSERT INTO without_rowid_store VALUES (?, ?)",
        [("a", "alpha"), ("b", None), ("雪", "unicode")],
    )
    con.commit()
    con.close()


class CompiledSqlirTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="sqlir-test-")
        cls.root = Path(cls.temporary.name)
        cls.db = cls.root / "fixture.sqlite"
        cls.model_dir = cls.root / "model"
        create_fixture(cls.db)
        compile_model(cls.db, WORKBOOK, cls.model_dir)
        cls.model = CompiledModel(cls.model_dir, verify="full")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_compile_and_live_differential_verify(self) -> None:
        report = verify_model(
            self.model_dir,
            self.db,
            samples=50,
            texts=["Earth"],
            workbook=WORKBOOK,
        )
        self.assertTrue(report["passed"], report["errors"])
        self.assertTrue(all(result["passed"] for result in report["query_results"]))
        self.assertEqual(report["forward_results"][0]["max_abs_diff"], 0.0)

    def test_forward_matches_reference_script(self) -> None:
        output = self.root / "forward-reference.json"
        subprocess.run(
            [
                sys.executable,
                str(REFERENCE_SCRIPT),
                "--workbook",
                str(WORKBOOK),
                "--db",
                str(self.db),
                "--text",
                "Earth",
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        expected = json.loads(output.read_text(encoding="utf-8"))
        actual = self.model.forward("Earth")
        for key in (
            "output_vectors",
            "attention",
            "input_vectors",
            "operator_sha256",
            "nearest_workbook_char_diagnostic",
        ):
            self.assertEqual(actual[key], expected[key])

    def test_missing_lookup_returns_empty(self) -> None:
        self.assertEqual(
            self.model.lookup("eq:indexed_child.parent_id", 9999), []
        )

    def test_paginated_fixed_and_index_lookup(self) -> None:
        cases = [
            ("fixed:conical.Q", None),
            ("eq:indexed_child.parent_id", 2),
        ]
        for query_id, key in cases:
            unpaginated = self.model.lookup(query_id, key)
            pages = []
            count = self.model.lookup_count(query_id, key)
            for offset in range(0, count, 7):
                pages.extend(
                    self.model.lookup(query_id, key, offset=offset, limit=7)
                )
            self.assertEqual(pages, unpaginated)
            self.assertEqual(count, len(unpaginated))

    def test_lookup_rejects_negative_pagination(self) -> None:
        with self.assertRaises(ValueError):
            self.model.lookup("fixed:conical.Q", offset=-1)
        with self.assertRaises(ValueError):
            self.model.lookup("fixed:conical.Q", limit=-1)

    def test_without_rowid_table_uses_primary_key_for_row(self) -> None:
        self.assertEqual(
            self.model.row("without_rowid_store", "b"), ["b", None]
        )

    def test_full_verification_detects_corrupted_blob(self) -> None:
        corrupt_dir = self.root / "corrupt-model"
        shutil.copytree(self.model_dir, corrupt_dir)
        manifest = json.loads(
            (corrupt_dir / "manifest.json").read_text(encoding="utf-8")
        )
        descriptor = next(
            item for item in manifest["segments"].values() if item["nbytes"] > 0
        )
        blob_path = corrupt_dir / "blob.bin"
        with blob_path.open("r+b") as blob:
            blob.seek(descriptor["offset"])
            original = blob.read(1)
            blob.seek(descriptor["offset"])
            blob.write(bytes([original[0] ^ 0x01]))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            CompiledModel(corrupt_dir, verify="full")

    def test_json_lines_server_round_trips_each_operation(self) -> None:
        requests = [
            {"id": 1, "op": "forward", "args": {"text": "Earth"}},
            {
                "id": 2,
                "op": "lookup",
                "args": {"query_id": "eq:indexed_child.parent_id", "key": 2},
            },
            {
                "id": 6,
                "op": "count",
                "args": {"query_id": "eq:indexed_child.parent_id", "key": 2},
            },
            {"id": 3, "op": "row", "args": {"table": "lexeme", "rowid": 1}},
            {"id": 4, "op": "list_queries", "args": {}},
            {"id": 5, "op": "manifest", "args": {}},
        ]
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "sqlir.serve",
                "--model",
                str(self.model_dir),
            ],
            input="".join(json.dumps(item) + "\n" for item in requests),
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        responses = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual([response["id"] for response in responses], [1, 2, 6, 3, 4, 5])
        self.assertTrue(all(response["ok"] for response in responses))
        self.assertEqual(responses[0]["result"]["input"], "Earth")
        self.assertIsInstance(responses[1]["result"], list)
        self.assertEqual(responses[2]["result"], len(responses[1]["result"]))
        self.assertIsInstance(responses[3]["result"], list)
        self.assertTrue(responses[4]["result"])
        self.assertIn("segments", responses[5]["result"])


if __name__ == "__main__":
    unittest.main()
