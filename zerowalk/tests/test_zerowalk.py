"""Software fixtures and canonical-workbook tests for the zero-walk package."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from zerowalk.certify import certify
from zerowalk.data import AggregatedRows
from zerowalk.discover import build_merge_trees, prepare_discovery_data
from zerowalk.__main__ import _write_model, build_parser
from zerowalk.readout import (
    fit_readout,
    scores_for,
    top1_accuracy,
    weighted_loss,
)
from zerowalk.walk import line_argmin, walk_edge, walk_known
from zerowalk.workbook import load_workbook
from zerowalk import zero_walk


ROOT = Path(__file__).resolve().parents[2]


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


if __name__ == "__main__":
    unittest.main()
