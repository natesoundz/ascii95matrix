"""Software fixtures and canonical-workbook tests for the zero-walk package."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from zerowalk.certify import certify
from zerowalk.data import AggregatedRows, prepare_data
from zerowalk.discover import build_merge_trees, prepare_discovery_data
from zerowalk.__main__ import _write_model, build_parser
from zerowalk.readout import (
    _class_block_proposals,
    _class_coordinate_proposals,
    _color_features,
    fit_readout,
    scores_for,
    top1_accuracy,
    weighted_loss,
)
from zerowalk.walk import (
    line_argmin,
    line_argmin_many,
    walk_edge,
    walk_known,
)
from zerowalk.workbook import load_workbook
from zerowalk import zero_walk


ROOT = Path(__file__).resolve().parents[2]
COMBINED_PATH = ROOT / "zerowalk" / "zero_walk_combined.py"
COMBINED_SPEC = importlib.util.spec_from_file_location(
    "zero_walk_combined",
    COMBINED_PATH,
)
COMBINED = importlib.util.module_from_spec(COMBINED_SPEC)
COMBINED_SPEC.loader.exec_module(COMBINED)


class WalkTests(unittest.TestCase):
    def test_walk_known_exhaustive_parity_and_traces(self):
        values = np.arange(-150, 151, dtype=np.int64)
        starts, goals = np.meshgrid(values, values, indexing="ij")
        delta, moves = walk_known(starts, goals)
        self.assertTrue(np.array_equal(delta, goals - starts))
        for index, (start, goal, move_count) in enumerate(zip(
            starts.flat, goals.flat, moves.flat
        )):
            expected_delta, trace = zero_walk.walk(int(start), int(goal))
            self.assertEqual(int(delta.flat[index]), expected_delta)
            self.assertEqual(int(move_count), len(trace))

        pairs = [
            (int(values[(i * 17) % values.size]),
             int(values[(i * 31) % values.size]))
            for i in range(2000)
        ]
        trace_delta, trace_moves, traces = walk_known(
            np.asarray([pair[0] for pair in pairs]),
            np.asarray([pair[1] for pair in pairs]),
            return_traces=True,
        )
        for index, (start, goal) in enumerate(pairs):
            expected_delta, expected_trace = zero_walk.walk(start, goal)
            self.assertEqual(int(trace_delta[index]), expected_delta)
            self.assertEqual(int(trace_moves[index]), len(expected_trace))
            self.assertEqual(traces[index], expected_trace)

    def test_walk_edge_exhaustive_vectorized_and_probe_bound(self):
        thresholds = np.arange(4097, dtype=np.int64)
        edge, probes, unbounded = walk_edge(
            lambda indices, offsets: offsets <= thresholds[indices],
            thresholds.size,
        )
        self.assertTrue(np.array_equal(edge, thresholds))
        self.assertFalse(np.any(unbounded))
        expected_probes = np.asarray([
            2 * ((int(value) + 1).bit_length() - 1) + 1
            for value in thresholds
        ])
        self.assertTrue(np.array_equal(probes, expected_probes))

        edge, probes, unbounded = walk_edge(
            lambda indices, offsets: np.ones(indices.size, dtype=bool),
            3,
            max_k=4,
        )
        self.assertTrue(np.array_equal(edge, np.full(3, 31)))
        self.assertTrue(np.array_equal(probes, np.full(3, 5)))
        self.assertTrue(np.all(unbounded))

    def test_line_argmin_against_piecewise_linear_software_fixtures(self):
        fixtures = [
            [(-4, 13), (-1, 4), (3, 7)],
            [(-3, 4), (0, 0), (2, 5)],
            [(-2, 7), (1, 0), (4, 2)],
            [(-5, 20), (-1, 1), (2, 0), (6, 8)],
            [(-3, 0), (2, 0)],
        ]
        for coefficients in fixtures:
            def objective(t, coefficients=coefficients):
                return max(slope * t + intercept
                           for slope, intercept in coefficients)

            expected = min(
                range(-50, 51),
                key=lambda t: (objective(t), abs(t)),
            )
            actual, comparisons = line_argmin(objective)
            self.assertEqual(actual, expected)
            self.assertGreater(comparisons, 0)

        calls = {}

        def counted_objective(t):
            calls[t] = calls.get(t, 0) + 1
            return abs(t - 4)

        self.assertEqual(line_argmin(counted_objective)[0], 4)
        self.assertTrue(all(count == 1 for count in calls.values()))

    def test_line_argmin_many_matches_combined_selftest_grid(self):
        centers = []
        weights = []
        flatnesses = []
        for center in range(-150, 151, 7):
            for weight in (1, 3):
                for flatness in (0, 2):
                    centers.append(center)
                    weights.append(weight)
                    flatnesses.append(flatness)
        centers = np.asarray(centers, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.int64)
        flatnesses = np.asarray(flatnesses, dtype=np.int64)

        def L_many(lines, steps):
            return weights[lines] * np.maximum(
                0,
                np.abs(steps - centers[lines]) - flatnesses[lines],
            )

        offsets, comparisons = line_argmin_many(
            L_many,
            centers.size,
        )
        expected = []
        expected_comparisons = 0
        for center, weight, flatness in zip(
            centers,
            weights,
            flatnesses,
        ):
            def objective(t, center=center, weight=weight, flatness=flatness):
                return int(weight) * max(
                    0,
                    abs(t - int(center)) - int(flatness),
                )

            offset, used = line_argmin(objective)
            expected.append(offset)
            expected_comparisons += used
        self.assertTrue(np.array_equal(
            offsets,
            np.asarray(expected, dtype=np.int64),
        ))
        self.assertEqual(comparisons, expected_comparisons)


class WorkbookAndReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding = load_workbook(
            ROOT / "sqlir" / "conical437_real_workbook.json"
        )

    def test_workbook_encoding_and_profile_groups(self):
        encoding = self.encoding
        self.assertEqual(encoding.features.shape, (95, 94))
        self.assertEqual(encoding.boolean_features, 21)
        self.assertEqual(encoding.one_hot_features, 73)
        self.assertEqual(encoding.profile_table.shape, (26, 94))
        self.assertEqual(np.linalg.matrix_rank(encoding.features), 26)
        self.assertEqual(len(encoding.groups), 26)
        multi = {"".join(group) for group in encoding.groups if len(group) > 1}
        self.assertEqual(
            multi,
            {
                "!\"&'*;?@\\",
                "#%",
                "([{",
                ")]}",
                ",./:",
                "<>",
                "=|~",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "abcdefghijklmnopqrstuvwxyz",
            },
        )
        self.assertIn("Separating them requires binding E/R/M/V cells", encoding.audit_text)

    def test_readout_certification_and_positive_scale(self):
        encoding = self.encoding
        X = np.column_stack((
            np.ones(encoding.profile_table.shape[0], dtype=np.int64),
            encoding.profile_table,
        ))
        y = np.arange(encoding.profile_table.shape[0], dtype=np.int64)
        rows = AggregatedRows(
            X=X,
            y=y,
            counts=np.ones(y.size, dtype=np.int64),
            context_keys=np.arange(y.size, dtype=np.int64),
        )
        W, fit = fit_readout(
            rows,
            max_sweeps=20,
            progress=False,
            loss="hinge",
        )
        self.assertLessEqual(fit["final_loss"], fit["initial_loss"])
        self.assertGreater(
            fit["final_train_accuracy"],
            fit["initial_train_accuracy"],
        )
        first_sweep_accuracy = fit["train_top1_accuracy_per_sweep"][0]
        _, stopped = fit_readout(
            rows,
            max_sweeps=20,
            stop_acc=first_sweep_accuracy,
            progress=False,
            loss="hinge",
        )
        self.assertEqual(stopped["sweeps"], 1)
        self.assertEqual(stopped["stop_reason"], "stop_acc")
        parsed = build_parser().parse_args([
            "fit", "--workbook", "workbook.json", "--text", "text.txt",
            "--offsets", "1", "--out", "out", "--stop-acc",
            str(first_sweep_accuracy),
        ])
        self.assertEqual(parsed.stop_acc, first_sweep_accuracy)
        self.assertEqual(parsed.loss, "softmax")
        self.assertEqual(parsed.unit_bits, 4)

        softmax_W, softmax_fit = fit_readout(
            rows,
            max_sweeps=50,
            progress=False,
        )
        self.assertEqual(softmax_fit["loss"], "softmax")
        self.assertEqual(
            top1_accuracy(scores_for(rows, softmax_W), y, rows.counts),
            1.0,
        )

        self.assertEqual(
            top1_accuracy(scores_for(rows, W), y, rows.counts),
            fit["final_train_accuracy"],
        )
        self.assertEqual(
            weighted_loss(scores_for(rows, W), y, rows.counts),
            fit["final_loss"],
        )
        self.assertTrue(
            np.array_equal(
                np.argmax(scores_for(rows, W), axis=1),
                np.argmax(scores_for(rows, 7 * W), axis=1),
            )
        )

        slack_up, slack_down, _ = certify(rows, W, max_k=16, progress=False)
        base_predictions = np.argmax(scores_for(rows, W), axis=1)
        checked = 0
        for class_index, feature_index in (
            (0, 0), (1, 1), (7, 2), (13, 3), (25, 10),
            (94, 0), (52, 20),
        ):
            for direction, slack_matrix in ((1, slack_up), (-1, slack_down)):
                slack = int(slack_matrix[class_index, feature_index])
                if slack < 0:
                    continue
                within = W.copy()
                within[class_index, feature_index] += direction * slack
                self.assertTrue(np.array_equal(
                    np.argmax(scores_for(rows, within), axis=1),
                    base_predictions,
                ))
                beyond = W.copy()
                beyond[class_index, feature_index] += direction * (slack + 1)
                self.assertTrue(np.any(
                    np.argmax(scores_for(rows, beyond), axis=1)
                    != base_predictions
                ))
                checked += 1
                if checked >= 3:
                    break
            if checked >= 3:
                break
        self.assertGreaterEqual(checked, 3)

        with TemporaryDirectory() as output_dir:
            _write_model(
                Path(output_dir),
                W,
                encoding.feature_names,
                encoding.profile_table,
                encoding.profile_id,
                offsets=1,
            )
            with np.load(Path(output_dir) / "model.npz") as model:
                self.assertEqual(model["feature_names"].size, W.shape[1])
                self.assertEqual(model["feature_names"][0], "bias")
                self.assertTrue(
                    str(model["feature_names"][1]).startswith("offset_-1:")
                )

    def test_discover_merge_tree_costs_on_readme_software_fixture(self):
        dataset = prepare_discovery_data(
            "text",
            ROOT / "README.md",
            offsets=1,
            encoding=self.encoding,
            progress=False,
        )
        W, _ = fit_readout(
            dataset.train,
            max_sweeps=1,
            progress=False,
            loss="softmax",
        )
        trees = build_merge_trees(
            dataset.train,
            W,
            self.encoding,
            dataset.occurrence_chars,
        )
        self.assertEqual(len(dataset.occurrence_chars), 78)
        columns = {
            name: index for index, name in enumerate(dataset.feature_names)
        }
        for tree in trees:
            self.assertEqual(
                len(tree["merges"]),
                len(tree["members"]) - 1,
            )
            effective = W.copy()
            for merge in tree["merges"]:
                before = dataset.train.X @ effective.T
                old_correct = int(np.dot(
                    dataset.train.counts,
                    np.argmax(before, axis=1) == dataset.train.y,
                ))
                for char in merge["moved_cluster"]:
                    source = columns[f"occurrence.o1.{char}"]
                    target = columns[
                        f"occurrence.o1.{merge['kept_representative']}"
                    ]
                    effective[:, source] = effective[:, target]
                after = dataset.train.X @ effective.T
                new_correct = int(np.dot(
                    dataset.train.counts,
                    np.argmax(after, axis=1) == dataset.train.y,
                ))
                self.assertEqual(
                    merge["cost_positions"],
                    old_correct - new_correct,
                )
                self.assertEqual(
                    merge["train_accuracy_after"],
                    new_correct / dataset.train.positions,
                )

        parsed = build_parser().parse_args([
            "discover", "--workbook", "workbook.json", "--text", "text.txt",
            "--out", "out",
        ])
        self.assertEqual(parsed.offsets, 1)
        self.assertEqual(parsed.loss, "softmax")
        self.assertEqual(parsed.unit_bits, 4)
        self.assertEqual(parsed.tolerance, 0.0)


class CombinedScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding = load_workbook(
            ROOT / "sqlir" / "conical437_real_workbook.json"
        )
        cls.rows = prepare_data(
            "text",
            ROOT / "README.md",
            1,
            cls.encoding,
        ).train

    def test_original_script_hash_is_unchanged(self):
        digest = hashlib.sha256(
            (ROOT / "zerowalk" / "zero_walk.py").read_bytes()
        ).hexdigest()
        self.assertEqual(
            digest,
            "74a10c77c00e3158907d885a47879237ad8cf527f0b59dccac985dd0c64d3efe",
        )

    def test_combined_selftest_software_fixture(self):
        self.assertEqual(
            COMBINED.selftest(argparse.Namespace(range=20)),
            0,
        )
        with self.assertRaisesRegex(ValueError, "at least 3"):
            COMBINED._off_label_top_two(
                np.zeros((1, 2), dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                2,
            )

    def test_real_text_readout_and_certify_parity(self):
        softmax_weights = None
        for loss in ("softmax", "hinge"):
            with self.subTest(loss=loss):
                package_W, package_stats = fit_readout(
                    self.rows,
                    max_sweeps=2,
                    progress=False,
                    loss=loss,
                )
                combined_W, combined_stats = COMBINED.fit_readout(
                    self.rows.X,
                    self.rows.y,
                    self.rows.counts,
                    95,
                    max_sweeps=2,
                    progress=False,
                    loss=loss,
                )
                self.assertTrue(np.array_equal(package_W, combined_W))
                self.assertEqual(
                    package_stats["loss_per_sweep"],
                    combined_stats["loss_per_sweep"],
                )
                self.assertEqual(
                    package_stats["train_top1_accuracy_per_sweep"],
                    combined_stats["train_top1_accuracy_per_sweep"],
                )
                self.assertEqual(
                    package_stats["stop_reason"],
                    combined_stats["stop_reason"],
                )
                if loss == "softmax":
                    softmax_weights = package_W

        package_up, package_down, package_summary = certify(
            self.rows,
            softmax_weights,
            progress=False,
        )
        combined_up, combined_down, combined_summary = COMBINED.certify(
            self.rows.X,
            self.rows.y,
            self.rows.counts,
            softmax_weights,
            progress=False,
        )
        self.assertTrue(np.array_equal(package_up, combined_up))
        self.assertTrue(np.array_equal(package_down, combined_down))
        self.assertEqual(package_summary, combined_summary)

    def test_fit_cli_round_trip_on_real_text_rows(self):
        rows_json = {
            "X": self.rows.X.tolist(),
            "y": self.rows.y.tolist(),
            "counts": self.rows.counts.tolist(),
            "classes": 95,
        }
        with TemporaryDirectory() as directory:
            rows_path = Path(directory) / "rows.json"
            model_path = Path(directory) / "model.json"
            rows_path.write_text(json.dumps(rows_json), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMBINED_PATH),
                    "fit",
                    "--rows",
                    str(rows_path),
                    "--out",
                    str(model_path),
                    "--max-sweeps",
                    "1",
                    "--certify",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("fit: softmax", completed.stdout)
            model = json.loads(model_path.read_text(encoding="utf-8"))
            expected_W, _ = COMBINED.fit_readout(
                self.rows.X,
                self.rows.y,
                self.rows.counts,
                95,
                max_sweeps=1,
                progress=False,
            )
            actual_W = np.asarray(
                model["W"]["data"],
                dtype=np.int64,
            ).reshape(model["W"]["shape"])
            self.assertTrue(np.array_equal(expected_W, actual_W))
            self.assertIsNotNone(model["certify"])

    def test_joint_worker_determinism_and_monotone_loss(self):
        serial_W, serial_stats = fit_readout(
            self.rows,
            max_sweeps=2,
            progress=False,
            strategy="joint",
            workers=1,
        )
        parallel_W, parallel_stats = fit_readout(
            self.rows,
            max_sweeps=2,
            progress=False,
            strategy="joint",
            workers=2,
        )
        self.assertTrue(np.array_equal(serial_W, parallel_W))
        ignored = {"wall_time_seconds", "workers"}
        self.assertEqual(
            {key: value for key, value in serial_stats.items() if key not in ignored},
            {key: value for key, value in parallel_stats.items() if key not in ignored},
        )

        _, three_sweep_stats = fit_readout(
            self.rows,
            max_sweeps=3,
            progress=False,
            strategy="joint",
            workers=1,
        )
        losses = three_sweep_stats["loss_per_sweep"]
        self.assertLessEqual(losses[0], three_sweep_stats["initial_loss"])
        self.assertTrue(all(
            later <= earlier for earlier, later in zip(losses, losses[1:])
        ))

    def test_fit_cli_parses_joint_strategy_and_workers(self):
        parsed = build_parser().parse_args([
            "fit",
            "--workbook",
            "workbook.json",
            "--text",
            "text.txt",
            "--offsets",
            "1",
            "--out",
            "out",
            "--strategy",
            "joint",
            "--workers",
            "2",
        ])
        self.assertEqual(parsed.strategy, "joint")
        self.assertEqual(parsed.workers, 2)

    def test_blocks_coloring_and_class_proposal_parity(self):
        color_groups = _color_features(self.rows.X)
        supports = self.rows.X != 0
        for group in color_groups:
            self.assertTrue(np.all(
                supports[:, group].sum(axis=1) <= 1
            ))
        self.assertEqual(
            [group.tolist() for group in color_groups],
            [group.tolist() for group in _color_features(self.rows.X)],
        )

        coordinate_weights, _ = fit_readout(
            self.rows,
            max_sweeps=1,
            progress=False,
            strategy="coordinate",
        )
        score_states = (
            ("zero-W", np.zeros((self.rows.y.size, 95), dtype=np.int64)),
            ("one-coordinate-sweep", self.rows.X @ coordinate_weights.T),
        )
        mismatch_counts = {
            (state, loss): 0
            for state, _ in score_states
            for loss in ("hinge", "softmax")
        }
        compared = {key: 0 for key in mismatch_counts}
        for state, scores in score_states:
            for loss in ("hinge", "softmax"):
                for class_index in range(4):
                    expected, *_ = _class_coordinate_proposals(
                        self.rows.X,
                        self.rows.y,
                        self.rows.counts,
                        scores.copy(),
                        class_index,
                        loss,
                        4,
                        40,
                    )
                    actual, *_ = _class_block_proposals(
                        self.rows.X,
                        self.rows.y,
                        self.rows.counts,
                        scores.copy(),
                        class_index,
                        loss,
                        4,
                        40,
                        color_groups,
                    )
                    for group in color_groups:
                        key = (state, loss)
                        mismatch_counts[key] += int(np.count_nonzero(
                            expected[group] != actual[group]
                        ))
                        compared[key] += group.size

        for state, _ in score_states:
            self.assertEqual(mismatch_counts[(state, "hinge")], 0)
        print("block proposal offset mismatches:")
        for state, _ in score_states:
            print(
                f"  {state}: "
                f"hinge={mismatch_counts[(state, 'hinge')]}/"
                f"{compared[(state, 'hinge')]}, "
                f"softmax={mismatch_counts[(state, 'softmax')]}/"
                f"{compared[(state, 'softmax')]}"
            )

    def test_blocks_softmax_loss_is_nonincreasing(self):
        _, stats = fit_readout(
            self.rows,
            max_sweeps=3,
            progress=False,
            strategy="blocks",
        )
        losses = stats["loss_per_sweep"]
        self.assertLessEqual(losses[0], stats["initial_loss"])
        self.assertTrue(all(
            later <= earlier for earlier, later in zip(losses, losses[1:])
        ))
        self.assertGreater(stats["colors"], 0)
        self.assertGreater(stats["max_color_size"], 0)
        _, hinge_stats = fit_readout(
            self.rows,
            max_sweeps=3,
            progress=False,
            loss="hinge",
            strategy="blocks",
        )
        hinge_losses = hinge_stats["loss_per_sweep"]
        self.assertLessEqual(hinge_losses[0], hinge_stats["initial_loss"])
        self.assertTrue(all(
            later <= earlier
            for earlier, later in zip(hinge_losses, hinge_losses[1:])
        ))
        with self.assertRaisesRegex(ValueError, "workers must be 1"):
            fit_readout(
                self.rows,
                max_sweeps=0,
                progress=False,
                strategy="blocks",
                workers=2,
            )

    def test_fit_cli_parses_blocks_strategy(self):
        parsed = build_parser().parse_args([
            "fit",
            "--workbook",
            "workbook.json",
            "--text",
            "text.txt",
            "--offsets",
            "1",
            "--out",
            "out",
            "--strategy",
            "blocks",
        ])
        self.assertEqual(parsed.strategy, "blocks")


if __name__ == "__main__":
    unittest.main()
