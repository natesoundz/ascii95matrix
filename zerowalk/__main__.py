"""Command-line interface for workbook audits and zero-walk fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .certify import certify
from .data import PreparedData, prepare_data
from .discover import (
    build_merge_trees,
    legacy_reference_matches,
    partition_metrics,
    prepare_discovery_data,
)
from .readout import fit_readout, scores_for, top1_accuracy
from .workbook import DEFAULT_REGISTRY, load_workbook


DEFAULT_LEGACY_REFERENCE = (
    Path(__file__).resolve().parent.parent
    / "legacy"
    / "ASCII95_254_BINARY_MATRIX.tsv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def _class_counts(rows) -> np.ndarray:
    totals = np.zeros(95, dtype=np.int64)
    np.add.at(totals, rows.y, rows.counts)
    return totals


def _feature_limited_bound(data: PreparedData) -> dict:
    train = data.train
    test = data.test
    global_counts = _class_counts(train)
    majority = int(np.argmax(global_counts))
    context_winners: dict[int, int] = {}
    train_correct = 0
    for context in np.unique(train.context_keys):
        mask = train.context_keys == context
        totals = np.zeros(95, dtype=np.int64)
        np.add.at(totals, train.y[mask], train.counts[mask])
        winner = int(np.argmax(totals))
        context_winners[int(context)] = winner
        train_correct += int(totals[winner])
    test_predictions = np.fromiter(
        (context_winners.get(int(key), majority) for key in test.context_keys),
        dtype=np.int64,
        count=test.context_keys.size,
    )
    test_correct = int(np.dot(test.counts, test_predictions == test.y))
    unseen = sum(int(key) not in context_winners for key in np.unique(test.context_keys))
    return {
        "train_top1_accuracy": _accuracy(train_correct, train.positions),
        "test_top1_accuracy": _accuracy(test_correct, test.positions),
        "distinct_train_contexts": len(context_winners),
        "unseen_test_contexts": unseen,
        "fallback_class_index": majority,
    }


def _majority_metrics(data: PreparedData) -> dict:
    totals = _class_counts(data.train)
    majority = int(np.argmax(totals))
    train_correct = int(totals[majority])
    test_correct = int(
        np.dot(data.test.counts, data.test.y == majority)
    )
    return {
        "class_index": majority,
        "character": chr(32 + majority),
        "train_top1_accuracy": _accuracy(train_correct, data.train.positions),
        "test_top1_accuracy": _accuracy(test_correct, data.test.positions),
    }


def _baseline_metrics(data: PreparedData) -> dict:
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(C=1.0, max_iter=1000)
    started = time.monotonic()
    classifier.fit(
        data.train.X[:, 1:],
        data.train.y,
        sample_weight=data.train.counts,
    )
    fit_time = time.monotonic() - started
    train_predictions = classifier.predict(data.train.X[:, 1:])
    test_predictions = classifier.predict(data.test.X[:, 1:])
    return {
        "train_top1_accuracy": _accuracy(
            int(np.dot(data.train.counts, train_predictions == data.train.y)),
            data.train.positions,
        ),
        "test_top1_accuracy": _accuracy(
            int(np.dot(data.test.counts, test_predictions == data.test.y)),
            data.test.positions,
        ),
        "fit_time_seconds": fit_time,
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 1000,
    }


def _write_model(
    out_dir: Path,
    W: np.ndarray,
    feature_names: tuple[str, ...],
    profile_table: np.ndarray,
    profile_id: np.ndarray,
    offsets: int,
    slack_up: np.ndarray | None = None,
    slack_down: np.ndarray | None = None,
) -> None:
    model_feature_names = ("bias",) + tuple(
        f"offset_-{offset}:{name}"
        for offset in range(1, offsets + 1)
        for name in feature_names
    )
    arrays = {
        "W": W,
        "feature_names": np.asarray(model_feature_names, dtype=np.str_),
        "profile_table": profile_table,
        "profile_id": profile_id,
        "offsets": np.asarray(offsets, dtype=np.int64),
    }
    if slack_up is not None and slack_down is not None:
        arrays["slack_up"] = slack_up
        arrays["slack_down"] = slack_down
    np.savez(out_dir / "model.npz", **arrays)


def _audit(args: argparse.Namespace) -> int:
    encoding = load_workbook(args.workbook)
    print(encoding.audit_text)
    return 0


def _fit(args: argparse.Namespace) -> int:
    workbook_path = Path(args.workbook)
    corpus_path = Path(args.text if args.text else args.records)
    mode = "text" if args.text else "records"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoding = load_workbook(workbook_path)
    data = prepare_data(mode, corpus_path, args.offsets, encoding)
    W, fit_stats = fit_readout(
        data.train,
        max_sweeps=args.max_sweeps,
        progress=True,
        stop_acc=args.stop_acc,
        patience=args.patience,
        loss=args.loss,
        unit_bits=args.unit_bits,
        strategy=args.strategy,
        workers=args.workers,
    )
    train_scores = scores_for(data.train, W)
    test_scores = scores_for(data.test, W)
    max_abs_weight = int(np.abs(W).max(initial=0))
    metrics = {
        "walk_readout": {
            "train_top1_accuracy": top1_accuracy(
                train_scores, data.train.y, data.train.counts
            ),
            "test_top1_accuracy": top1_accuracy(
                test_scores, data.test.y, data.test.counts
            ),
        },
        "feature_limited_bound": _feature_limited_bound(data),
        "majority_class": _majority_metrics(data),
        "walk_readout_fit": fit_stats,
        "max_abs_weight": max_abs_weight,
        "weight_bits_signed": int(np.ceil(np.log2(max_abs_weight + 1))) + 1,
        "nnz_weights": int(np.count_nonzero(W)),
    }
    if args.baselines:
        metrics["sklearn_logistic_regression"] = _baseline_metrics(data)

    slack_up = None
    slack_down = None
    if args.certify:
        slack_up, slack_down, certify_stats = certify(data.train, W)
        metrics["certify"] = certify_stats

    _write_model(
        out_dir,
        W,
        encoding.feature_names,
        encoding.profile_table,
        encoding.profile_id,
        args.offsets,
        slack_up,
        slack_down,
    )

    corpus_name = corpus_path.name
    report = {
        "method": "integer coordinate search; comparison-only zero walk",
        "walk_edge_probe_bound": (
            "For finite edge T, probes = "
            "2 * floor(log2(T + 1)) + 1."
        ),
        "inputs": {
            "workbook": {
                "basename": workbook_path.name,
                "sha256": _sha256(workbook_path),
            },
            "registry_dimensions": {
                "basename": DEFAULT_REGISTRY.name,
                "sha256": _sha256(DEFAULT_REGISTRY),
            },
            "corpus": {
                "basename": corpus_name,
                "sha256": _sha256(corpus_path),
            },
        },
        "data": {
            "accepted_records": data.accepted_records,
            "rejected_records": data.rejected_records,
            "train_records": data.train_records,
            "test_records": data.test_records,
            "train_positions": data.train.positions,
            "test_positions": data.test.positions,
            "distinct_train_rows": int(data.train.y.size),
            "distinct_test_rows": int(data.test.y.size),
        },
        "encoding": {
            "features": int(encoding.features.shape[1]),
            "boolean_features": encoding.boolean_features,
            "one_hot_features": encoding.one_hot_features,
            "profile_count": int(encoding.profile_table.shape[0]),
            "profile_rank": int(np.linalg.matrix_rank(encoding.features)),
            "groups": ["".join(group) for group in encoding.groups],
            "audit": (
                "These groups are indistinguishable from defined evidence. "
                "Separating them requires binding E/R/M/V cells. Do not invent "
                "those values."
            ),
        },
        "offsets": args.offsets,
        "loss": args.loss,
        "unit_bits": args.unit_bits,
        "strategy": args.strategy,
        "workers": args.workers,
        "metrics": metrics,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"records accepted/rejected: {data.accepted_records}/"
        f"{data.rejected_records}; positions train/test: "
        f"{data.train.positions}/{data.test.positions}"
    )
    print(
        f"walk accuracy train/test: "
        f"{metrics['walk_readout']['train_top1_accuracy']}/"
        f"{metrics['walk_readout']['test_top1_accuracy']}; "
        f"stop={fit_stats['stop_reason']}; loss={fit_stats['final_loss']}"
    )
    print(f"report: {report_path}")
    return 0


def _discover(args: argparse.Namespace) -> int:
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative value")
    workbook_path = Path(args.workbook)
    corpus_path = Path(args.text if args.text else args.records)
    mode = "text" if args.text else "records"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoding = load_workbook(workbook_path)
    dataset = prepare_discovery_data(
        mode,
        corpus_path,
        args.offsets,
        encoding,
    )
    W, fit_stats = fit_readout(
        dataset.train,
        max_sweeps=args.max_sweeps,
        progress=True,
        patience=args.patience,
        loss=args.loss,
        unit_bits=args.unit_bits,
    )
    train_scores = scores_for(dataset.train, W)
    test_scores = scores_for(dataset.test, W)
    trees = build_merge_trees(
        dataset.train,
        W,
        encoding,
        dataset.occurrence_chars,
    )
    partition = partition_metrics(
        dataset,
        W,
        trees,
        args.offsets,
        args.tolerance,
    )
    legacy = legacy_reference_matches(trees, DEFAULT_LEGACY_REFERENCE)
    legacy["sha256"] = _sha256(DEFAULT_LEGACY_REFERENCE)
    top_splits = [
        {
            "group": tree["group"],
            "cluster_A": tree["merges"][-1]["cluster_A"],
            "cluster_B": tree["merges"][-1]["cluster_B"],
            "kept_representative": tree["merges"][-1][
                "kept_representative"
            ],
        }
        for tree in trees
    ]
    report = {
        "method": "occurrence-bound integer search and greedy agglomerative merging",
        "inputs": {
            "workbook": {
                "basename": workbook_path.name,
                "sha256": _sha256(workbook_path),
            },
            "registry_dimensions": {
                "basename": DEFAULT_REGISTRY.name,
                "sha256": _sha256(DEFAULT_REGISTRY),
            },
            "corpus": {
                "basename": corpus_path.name,
                "sha256": _sha256(corpus_path),
            },
        },
        "data": {
            "accepted_records": dataset.accepted_records,
            "rejected_records": dataset.rejected_records,
            "train_records": dataset.train_records,
            "test_records": dataset.test_records,
            "train_positions": dataset.train.positions,
            "test_positions": dataset.test.positions,
            "distinct_train_rows": int(dataset.train.y.size),
            "distinct_test_rows": int(dataset.test.y.size),
            "distinct_train_contexts": int(np.unique(
                dataset.train.context_keys,
                axis=0,
            ).shape[0]),
            "distinct_test_contexts": int(np.unique(
                dataset.test.context_keys,
                axis=0,
            ).shape[0]),
        },
        "offsets": args.offsets,
        "loss": args.loss,
        "unit_bits": args.unit_bits,
        "occurrence_character_count": len(dataset.occurrence_chars),
        "occurrence_characters": "".join(dataset.occurrence_chars),
        "fit": {
            "train_top1_accuracy": top1_accuracy(
                train_scores,
                dataset.train.y,
                dataset.train.counts,
            ),
            "test_top1_accuracy": top1_accuracy(
                test_scores,
                dataset.test.y,
                dataset.test.counts,
            ),
            "stats": fit_stats,
        },
        "merge_trees": trees,
        "tolerance_partition": partition,
        "top_splits": top_splits,
        "external_reference": legacy,
    }
    report_path = out_dir / "discover.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"discover offsets={args.offsets}, loss={args.loss}, "
        f"tolerance={args.tolerance}: train/test accuracy "
        f"{partition['train_top1_accuracy']}/"
        f"{partition['test_top1_accuracy']}"
    )
    for group in partition["clusters_by_group"]:
        print(f"partition {group['group']!r}: {group['clusters']}")
    for split in top_splits:
        if split["group"] in {
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "abcdefghijklmnopqrstuvwxyz",
        }:
            print(
                f"top split {split['group']}: {split['cluster_A']} | "
                f"{split['cluster_B']}"
            )
    print(f"report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m zerowalk")
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="audit workbook encoding")
    audit_parser.add_argument("--workbook", required=True)
    audit_parser.set_defaults(run=_audit)

    fit_parser = commands.add_parser("fit", help="fit integer walk readout")
    fit_parser.add_argument("--workbook", required=True)
    input_group = fit_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text")
    input_group.add_argument("--records")
    fit_parser.add_argument("--offsets", type=int, required=True)
    fit_parser.add_argument("--out", required=True)
    fit_parser.add_argument("--max-sweeps", type=int, default=50)
    fit_parser.add_argument("--stop-acc", type=float, default=None)
    fit_parser.add_argument("--patience", type=int, default=3)
    fit_parser.add_argument(
        "--loss",
        choices=("softmax", "hinge"),
        default="softmax",
    )
    fit_parser.add_argument("--unit-bits", type=int, default=4)
    fit_parser.add_argument(
        "--strategy",
        choices=("coordinate", "joint", "blocks"),
        default="coordinate",
    )
    fit_parser.add_argument("--workers", type=int, default=1)
    fit_parser.add_argument("--baselines", action="store_true")
    fit_parser.add_argument("--certify", action="store_true")
    fit_parser.set_defaults(run=_fit)

    discover_parser = commands.add_parser(
        "discover",
        help="fit occurrence-bound features and build merge trees",
    )
    discover_parser.add_argument("--workbook", required=True)
    discover_input_group = discover_parser.add_mutually_exclusive_group(
        required=True
    )
    discover_input_group.add_argument("--text")
    discover_input_group.add_argument("--records")
    discover_parser.add_argument("--offsets", type=int, default=1)
    discover_parser.add_argument("--out", required=True)
    discover_parser.add_argument("--tolerance", type=float, default=0.0)
    discover_parser.add_argument("--loss", choices=("softmax", "hinge"), default="softmax")
    discover_parser.add_argument("--unit-bits", type=int, default=4)
    discover_parser.add_argument("--max-sweeps", type=int, default=50)
    discover_parser.add_argument("--patience", type=int, default=3)
    discover_parser.set_defaults(run=_discover)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
