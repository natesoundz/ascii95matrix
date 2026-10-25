"""TEST-ONLY adapter fixture tests; fixture data is never production input."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sqlir.runtime import CompiledModel
from sqlir.sql_adapter import SQLAdapter
from sqlir.tests.test_sqlir import ROOT, WORKBOOK, create_fixture
from sqlir.verify import verify_model
from sqlir.weights import DirectWeights, RubikWeights, export_rubik
from sqlir.model_adapter import ModelAdapter


RUBIK_MODULE = os.environ.get("SQLIR_RUBIK_MODULE")


class SQLAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="sqlir-adapter-test-")
        cls.root = Path(cls.temporary.name)
        cls.db = cls.root / "fixture.sqlite"
        cls.model_dir = cls.root / "model"
        create_fixture(cls.db)
        from sqlir.compile import compile_model

        compile_model(cls.db, WORKBOOK, cls.model_dir)
        cls.model = CompiledModel(cls.model_dir, verify="full")
        cls.adapter = SQLAdapter(str(cls.model_dir), seed=7)
        cls.adapter.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_adapter_is_concrete(self) -> None:
        self.assertFalse(inspect.isabstract(SQLAdapter))
        self.assertIsInstance(self.adapter, ModelAdapter)
        self.assertEqual(self.adapter.get_vocab_size(), self.model.manifest["lm"]["vocab_size"])

    def test_bigram_and_collocation_rows_match_fixture_counts(self) -> None:
        earth = self.adapter._id_for_word("earth")
        sun = self.adapter._id_for_word("sun")
        world = self.adapter._id_for_word("world")
        cols, counts = self.adapter._csr_row("bigram", earth)
        actual = {int(col): int(value) for col, value in zip(cols, counts)}
        self.assertEqual(actual[sun], 2)
        cols, weights = self.adapter._csr_row("colloc", earth)
        actual_colloc = {int(col): float(value) for col, value in zip(cols, weights)}
        self.assertEqual(actual_colloc[sun], 3.0)
        self.assertEqual(actual_colloc[world], 1.0)

    def test_definitions_and_missing_definition_are_explicit(self) -> None:
        result = self.adapter.run("define earth")
        self.assertEqual(result["procedure"], "define")
        self.assertEqual(
            result["text"],
            "1. the planet on which we live\n2. the third planet from the sun",
        )
        missing = self.adapter.run("define unlistedword")
        self.assertEqual(
            missing["text"], "No definition for unlistedword in sense.definition"
        )

    def test_temperature_zero_continuation_is_deterministic_and_attested(self) -> None:
        first = self.adapter.run("the earth", max_tokens=8, temperature=0.0)
        second = self.adapter.run("the earth", max_tokens=8, temperature=0.0)
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(first["tokens"], second["tokens"])
        self.assertTrue(first["lexicon_check"]["passed"])
        self.assertTrue(
            all(
                item["bigram"] > 0 or item["colloc"] > 0
                for item in first["lexicon_check"]["transitions"]
            )
        )

    def test_residual_and_logit_writes_apply_on_next_forward(self) -> None:
        tokens = self.adapter.tokenize("earth sun")
        base_logits, _ = self.adapter.forward(tokens)
        base_attention = self.adapter.read_attention_weights(0)
        perturbation = np.vstack(
            [np.linspace(-100.0, 100.0, 437), np.linspace(75.0, -125.0, 437)]
        )
        self.adapter.write_residual_stream(0, perturbation)
        changed_logits, _ = self.adapter.forward(tokens)
        changed_attention = self.adapter.read_attention_weights(0)
        self.assertFalse(np.array_equal(base_attention, changed_attention))
        self.assertFalse(np.array_equal(base_logits, changed_logits))

        finite = np.flatnonzero(np.isfinite(changed_logits))
        chosen = int(finite[-1])
        modification = np.zeros(self.adapter.get_vocab_size())
        modification[chosen] = 1_000_000.0
        self.adapter.modify_logits(modification)
        modified_logits, _ = self.adapter.forward(tokens)
        self.assertEqual(self.adapter.sample(modified_logits, temperature=0), chosen)

    def test_kv_cache_shape_and_readers_return_copies(self) -> None:
        tokens = self.adapter.tokenize("earth sun")
        self.adapter.forward(tokens)
        cache = self.adapter.read_kv_cache()
        self.assertEqual(cache[0]["key"].shape, (len(tokens), 437))
        cache[0]["key"][0, 0] += 10
        self.assertNotEqual(
            cache[0]["key"][0, 0], self.adapter.read_kv_cache()[0]["key"][0, 0]
        )
        self.assertEqual(self.adapter.read_attention_weights(0).shape, (len(tokens), len(tokens)))

    def test_scan_reports_compiled_evidence_and_attestation(self) -> None:
        scan = self.adapter.scan("earth sun nonesuch")
        earth = next(item for item in scan["words"] if item["word"] == "earth")
        self.assertEqual(
            earth["definitions"],
            ["the planet on which we live", "the third planet from the sun"],
        )
        self.assertEqual(
            earth["relations"],
            self.adapter._relation_values(self.adapter._id_for_word("earth")),
        )
        self.assertEqual(earth["next_by_bigram"][0], ["sun", 2])
        self.assertIn("nonesuch", scan["unknown"])
        self.assertTrue(scan["pairs"][0]["attested"])
        self.assertFalse(scan["pairs"][-1]["attested"])
        define = self.adapter.run("define earth")
        self.assertEqual(
            [line.split(". ", 1)[1] for line in define["text"].splitlines()],
            earth["definitions"],
        )
        related = self.adapter.run("related to earth")
        groups = {}
        for relation_type, target in earth["relations"]:
            groups.setdefault("null" if relation_type is None else relation_type, []).append(
                target
            )
        expected_related = "\n".join(
            f"{relation_type}: {', '.join(targets)}"
            for relation_type, targets in sorted(groups.items())
        )
        self.assertEqual(related["text"], expected_related)
        self.assertIn("scan", define)
        self.assertIn("groups", earth["workbook"])
        self.assertIn("unresolved_cells", earth["workbook"])

    def test_inflection_mappings_round_trip(self) -> None:
        scan = self.adapter.scan("suns sun")
        form = next(item for item in scan["words"] if item["word"] == "suns")
        base = next(item for item in scan["words"] if item["word"] == "sun")
        self.assertEqual(form["lemma_of"], ["sun"])
        self.assertIn("suns", base["inflections"])

    def test_fixture_verification_passes_and_corruption_fails(self) -> None:
        report = verify_model(self.model_dir, self.db, samples=50, workbook=WORKBOOK)
        self.assertTrue(report["passed"], report["errors"])
        corrupt_dir = self.root / "corrupt"
        shutil.copytree(self.model_dir, corrupt_dir)
        manifest = json.loads((corrupt_dir / "manifest.json").read_text())
        segment = manifest["lm"]["segments"]["bigram"]["value"]
        descriptor = manifest["segments"][segment]
        with (corrupt_dir / "blob.bin").open("r+b") as blob:
            blob.seek(descriptor["offset"])
            byte = blob.read(1)
            blob.seek(descriptor["offset"])
            blob.write(bytes([byte[0] ^ 1]))
        corrupted = verify_model(
            corrupt_dir,
            self.db,
            samples=2,
            workbook=WORKBOOK,
            report_path=self.root / "corrupt-verify-report.json",
        )
        self.assertFalse(corrupted["passed"])

    def test_serving_generate_round_trip(self) -> None:
        requests = [
            {"id": 1, "op": "generate", "args": {"prompt": "define earth"}},
            {"id": 2, "op": "scan", "args": {"text": "earth sun"}},
        ]
        process = subprocess.run(
            [sys.executable, "-m", "sqlir.serve", "--model", str(self.model_dir)],
            input="".join(json.dumps(request) + "\n" for request in requests),
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        responses = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertTrue(all(response["ok"] for response in responses), responses)
        self.assertEqual(responses[0]["result"]["procedure"], "define")
        self.assertEqual(
            responses[0]["result"]["text"].splitlines()[0],
            "1. the planet on which we live",
        )
        self.assertEqual(responses[1]["result"]["pairs"][0]["attested"], True)

    def test_direct_weight_source_is_bit_identical(self) -> None:
        expected = self.model.forward("Earth")
        actual = self.model.forward("Earth", weights=DirectWeights(self.model))
        self.assertEqual(actual, expected)

    def test_missing_rubik_module_has_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Could not import Rubik module"):
            RubikWeights(self.root / "missing-store", self.root / "missing-rubik.py")

    @unittest.skipUnless(RUBIK_MODULE, "SQLIR_RUBIK_MODULE is not configured")
    def test_rubik_weight_source_is_bit_identical_and_journals_reads(self) -> None:
        rubik_dir = self.root / "rubik"
        export_rubik(self.model_dir, rubik_dir, RUBIK_MODULE)
        rubik = RubikWeights(rubik_dir, RUBIK_MODULE)
        before = len(rubik.store.journal)
        expected = self.model.forward("Earth", weights=DirectWeights(self.model))
        actual = self.model.forward("Earth", weights=rubik)
        self.assertEqual(actual, expected)
        direct_adapter = SQLAdapter(str(self.model_dir), weights=DirectWeights(self.model))
        rubik_adapter = SQLAdapter(str(self.model_dir), weights=rubik)
        direct_logits, _ = direct_adapter.forward(direct_adapter.tokenize("earth sun"))
        rubik_logits, _ = rubik_adapter.forward(rubik_adapter.tokenize("earth sun"))
        np.testing.assert_array_equal(direct_logits, rubik_logits)
        np.testing.assert_array_equal(
            direct_adapter.read_attention_weights(0),
            rubik_adapter.read_attention_weights(0),
        )
        np.testing.assert_array_equal(
            direct_adapter.read_kv_cache()[0]["key"],
            rubik_adapter.read_kv_cache()[0]["key"],
        )
        self.assertGreater(len(rubik.store.journal), before)


if __name__ == "__main__":
    unittest.main()
