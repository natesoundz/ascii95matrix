"""Differential verification against live SQLite and the reference forward pass."""

from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import canonical_json, canonical_row, json_value
from .lm_compile import SPECIAL_WORDS, norm_word
from .runtime import CompiledModel


PAGE_SIZE = 10_000


def _sample_positions(count: int, samples: int) -> list[int]:
    if count == 0:
        return []
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if count <= samples:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, num=samples, dtype=np.int64).tolist()))


def _live_row_pages(con, query: dict, key: Any = None):
    if query["kind"] in {"fixed", "count"}:
        cursor = con.execute(query["sql"])
    else:
        if query["kind"] == "text":
            con.create_function("sqlir_lower", 1, lambda value: None if value is None else str(value).lower())
        if isinstance(key, dict) and set(key) == {"$b64"}:
            key = base64.b64decode(key["$b64"])
        elif len(query["key_columns"]) > 1:
            key = [
                base64.b64decode(value["$b64"])
                if isinstance(value, dict) and set(value) == {"$b64"}
                else value
                for value in key
            ]
        if len(query["key_columns"]) == 1:
            params = (key,)
        else:
            params = tuple(key)
        cursor = con.execute(query["sql"], params)
    while rows := cursor.fetchmany(PAGE_SIZE):
        yield [[json_value(value) for value in row] for row in rows]


def _canonical_rows(rows: list) -> bytes:
    return canonical_json(rows)


def _compare_paged(
    model: CompiledModel, con, query: dict, key: Any = None
) -> tuple[bool, int, list[int]]:
    query_id = query["id"]
    actual_count = model.lookup_count(query_id, key)
    expected_count = 0
    mismatched_offsets = []
    for expected_page in _live_row_pages(con, query, key):
        actual_page = model.lookup(
            query_id, key, offset=expected_count, limit=PAGE_SIZE
        )
        if _canonical_rows(actual_page) != _canonical_rows(expected_page):
            mismatched_offsets.append(expected_count)
        expected_count += len(expected_page)
    if expected_count != actual_count:
        mismatched_offsets.append(expected_count)
    return not mismatched_offsets, actual_count, mismatched_offsets


def _verify_forward(
    model: CompiledModel,
    db: Path,
    workbook: Path,
    script: Path,
    texts: list[str],
) -> tuple[list[dict], list[str]]:
    results = []
    errors = []
    for text in texts:
        with tempfile.TemporaryDirectory(prefix="sqlir-forward-") as directory:
            output_path = Path(directory) / "reference.json"
            command = [
                sys.executable,
                str(script),
                "--workbook",
                str(workbook),
                "--db",
                str(db),
                "--text",
                text,
                "--output",
                str(output_path),
            ]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode != 0:
                errors.append(
                    f"reference forward failed for {text!r}: "
                    f"{process.stderr.strip() or process.stdout.strip()}"
                )
                continue
            reference = json.loads(output_path.read_text(encoding="utf-8"))
            actual = model.forward(text)
            differences = {}
            max_abs_diff = 0.0
            for name in ("input_vectors", "attention", "output_vectors"):
                left = np.asarray(reference[name], dtype=np.float64)
                right = np.asarray(actual[name], dtype=np.float64)
                if left.shape != right.shape:
                    differences[name] = "shape mismatch"
                    max_abs_diff = float("inf")
                else:
                    difference = float(np.max(np.abs(left - right))) if left.size else 0.0
                    max_abs_diff = max(max_abs_diff, difference)
                    if not np.array_equal(left, right):
                        differences[name] = difference
            for name in ("operator_sha256", "nearest_workbook_char_diagnostic"):
                if reference[name] != actual[name]:
                    differences[name] = "exact value mismatch"
            results.append(
                {
                    "text": text,
                    "max_abs_diff": max_abs_diff,
                    "passed": not differences and max_abs_diff == 0.0,
                    "differences": differences,
                }
            )
            if differences or max_abs_diff != 0.0:
                errors.append(f"forward mismatch for {text!r}: {differences}")
    return results, errors


def _verify_lm(model: CompiledModel, db: Path) -> tuple[dict, list[str]]:
    from .sql_adapter import SQLAdapter

    errors = []
    adapter = SQLAdapter(str(model.directory))
    adapter.load()
    lm = model.manifest.get("lm", {})
    vocab_size = int(lm.get("vocab_size", 0))
    if vocab_size < len(SPECIAL_WORDS):
        return {"passed": False, "error": "LM vocabulary is missing"}, [
            "LM vocabulary is missing"
        ]
    positions = sorted(
        set(
            np.linspace(
                3, vocab_size - 1, num=min(50, vocab_size - 3), dtype=np.int64
            ).tolist()
        )
    )
    sample_words = {position: adapter._word_for_id(position) for position in positions}
    sample_ids = set(sample_words)
    uri = db.as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only=ON")

    with tempfile.TemporaryDirectory(prefix="sqlir-lm-verify-") as directory:
        vocab_db = sqlite3.connect(Path(directory) / "vocab.sqlite")
        vocab_db.execute("CREATE TABLE words (word TEXT PRIMARY KEY) WITHOUT ROWID")
        for name, source in lm.get("sources", {}).items():
            if not name.startswith("vocab:"):
                continue
            for (value,) in con.execute(source["sql"]):
                try:
                    word = norm_word(str(value)) if value is not None else None
                except (TypeError, ValueError):
                    continue
                if word is not None:
                    vocab_db.execute("INSERT OR IGNORE INTO words VALUES (?)", (word,))
        vocab_db.commit()
        live_vocab_size = int(vocab_db.execute("SELECT COUNT(*) FROM words").fetchone()[0]) + 3
        vocab_db.close()

    bigram_source = lm.get("sources", {}).get("bigrams:occurrence.tokens")
    live_bigrams: dict[int, dict[int, int]] = {token_id: {} for token_id in sample_ids}
    if bigram_source:
        sentinel = object()
        example = sentinel
        previous = None

        def add_bigram(source: int, target: int) -> None:
            if source in live_bigrams and source != 0 and target != 0:
                row = live_bigrams[source]
                row[target] = row.get(target, 0) + 1

        for example_id, _token_index, raw_word in con.execute(bigram_source["sql"]):
            if example_id != example:
                if previous not in (None, 0):
                    add_bigram(previous, 2)
                example = example_id
                previous = 1
            try:
                word = norm_word(str(raw_word)) if raw_word is not None else None
            except (TypeError, ValueError):
                word = None
            current = adapter._id_for_word(word) if word is not None else 0
            add_bigram(previous, current)
            previous = current
        if previous not in (None, 0):
            add_bigram(previous, 2)

    colloc_source = lm.get("sources", {}).get("colloc:collocation")
    live_colloc: dict[int, dict[int, float]] = {token_id: {} for token_id in sample_ids}
    if colloc_source:
        for left, right, frequency in con.execute(colloc_source["sql"]):
            try:
                left_word = norm_word(str(left)) if left is not None else None
                right_word = norm_word(str(right)) if right is not None else None
            except (TypeError, ValueError):
                continue
            if left_word is None or right_word is None:
                continue
            source_id = adapter._id_for_word(left_word)
            target_id = adapter._id_for_word(right_word)
            if source_id not in live_colloc or source_id == 0 or target_id == 0:
                continue
            weight = 1.0 if frequency is None else float(frequency)
            if not np.isfinite(weight):
                continue
            row = live_colloc[source_id]
            row[target_id] = row.get(target_id, 0.0) + weight

    live_definitions: dict[int, list[str]] = {token_id: [] for token_id in sample_ids}
    definition_source = lm.get("sources", {}).get("definitions")
    if definition_source:
        for word, text in con.execute(definition_source["sql"]):
            if word is None or text is None:
                continue
            try:
                normalized = norm_word(str(word))
            except ValueError:
                continue
            token_id = adapter._id_for_word(normalized)
            if token_id in live_definitions and str(text) not in live_definitions[token_id]:
                live_definitions[token_id].append(str(text))

    rows_checked = 0
    for token_id in positions:
        try:
            cols, values = adapter._csr_row("bigram", token_id)
            actual_bigram = {
                int(col): int(value) for col, value in zip(cols, values)
            }
            if actual_bigram != live_bigrams[token_id]:
                errors.append(f"LM bigram mismatch for {sample_words[token_id]!r}")
        except Exception as exc:
            errors.append(
                f"LM bigram read failed for {sample_words[token_id]!r}: {exc}"
            )
        try:
            cols, values = adapter._csr_row("colloc", token_id)
            actual_colloc = {
                int(col): float(value) for col, value in zip(cols, values)
            }
            expected_colloc = live_colloc[token_id]
            if actual_colloc.keys() != expected_colloc.keys() or any(
                actual_colloc[key] != expected_colloc[key] for key in actual_colloc
            ):
                errors.append(
                    f"LM collocation mismatch for {sample_words[token_id]!r}"
                )
        except Exception as exc:
            errors.append(
                f"LM collocation read failed for {sample_words[token_id]!r}: {exc}"
            )
        try:
            if adapter._definition_values(token_id) != live_definitions[token_id]:
                errors.append(f"LM definition mismatch for {sample_words[token_id]!r}")
        except Exception as exc:
            errors.append(
                f"LM definition read failed for {sample_words[token_id]!r}: {exc}"
            )
        rows_checked += 1

    con.close()
    vocab_matches = live_vocab_size == vocab_size
    if not vocab_matches:
        errors.append(
            f"LM vocabulary count mismatch: artifact={vocab_size}, live={live_vocab_size}"
        )
    first_word = adapter._word_for_id(3)
    procedure_results = []
    for prompt in (f"define {first_word}", f"related to {first_word}", first_word):
        try:
            result = adapter.run(prompt, max_tokens=8, temperature=0.0)
            passed = bool(result["lexicon_check"]["passed"])
            procedure_error = None
        except Exception as exc:
            passed = False
            procedure_error = str(exc)
        procedure_results.append(
            {
                "prompt": prompt,
                "procedure": (
                    result["procedure"] if procedure_error is None else None
                ),
                "lexicon_check_passed": passed,
                "error": procedure_error,
            }
        )
        if not passed:
            errors.append(f"LM lexicon check failed for {prompt!r}")
    result = {
        "passed": not errors,
        "vocab_size_artifact": vocab_size,
        "vocab_size_live": live_vocab_size,
        "vocab_size_matches": vocab_matches,
        "sampled_vocab_words": list(sample_words.values()),
        "sampled_rows_checked": rows_checked,
        "procedures": procedure_results,
    }
    return result, errors


def verify_model(
    model_dir: Path,
    db: Path,
    samples: int = 50,
    texts: list[str] | None = None,
    workbook: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    model_dir = model_dir.resolve()
    db = db.resolve()
    model = CompiledModel(model_dir, verify="none")
    workbook = (workbook or Path(model.manifest.get("workbook_path", ""))).resolve()
    if not workbook.is_file():
        workbook = Path(__file__).with_name("conical437_real_workbook.json")
    script = Path(__file__).with_name("conical_sql_ir_transformer.py")
    uri = db.as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only=ON")
    con.create_function(
        "sqlir_lower", 1, lambda value: None if value is None else str(value).lower()
    )
    query_results = []
    errors = []
    for query in model.list_queries():
        if query["kind"] in {"fixed", "count"}:
            passed, row_count, mismatches = _compare_paged(model, con, query)
            query_results.append(
                {
                    "id": query["id"],
                    "kind": query["kind"],
                    "sample_keys": "all",
                    "passed": passed,
                    "rows_compiled": row_count,
                    "mismatched_offsets": mismatches,
                }
            )
            if not passed:
                errors.append(f"query mismatch: {query['id']}")
            continue

        keys = model._query_keys(query)
        positions = _sample_positions(len(keys), samples)
        mismatched = []
        for position in positions:
            key = keys[position]
            passed, row_count, offsets = _compare_paged(model, con, query, key)
            if not passed:
                mismatched.append(
                    {
                        "position": position,
                        "key": key,
                        "rows_compiled": row_count,
                        "mismatched_offsets": offsets,
                    }
                )
        passed = not mismatched
        query_results.append(
            {
                "id": query["id"],
                "kind": query["kind"],
                "sample_keys": [keys[position] for position in positions],
                "passed": passed,
                "mismatches": mismatched,
            }
        )
        if mismatched:
            errors.append(f"query mismatch: {query['id']} keys={mismatched}")
    con.close()

    forward_results, forward_errors = _verify_forward(
        model, db, workbook, script, texts or []
    )
    errors.extend(forward_errors)
    lm_results, lm_errors = _verify_lm(model, db)
    errors.extend(lm_errors)
    try:
        model.verify_all()
        segments_passed = True
    except Exception as exc:
        segments_passed = False
        errors.append(f"segment verification failed: {exc}")
    report = {
        "model": str(model_dir),
        "db": str(db),
        "query_results": query_results,
        "forward_results": forward_results,
        "lm_results": lm_results,
        "all_segments_sha256_passed": segments_passed,
        "passed": not errors,
        "errors": errors,
    }
    report_path = report_path or model_dir / "verify_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Verification {'PASSED' if report['passed'] else 'FAILED'}: "
        f"{len(query_results)} queries, {len(forward_results)} forward cases, "
        f"{'all segments' if segments_passed else 'segment mismatch'}"
    )
    for item in forward_results:
        print(
            f"FORWARD {item['text']!r} max_abs_diff={item['max_abs_diff']} "
            f"passed={item['passed']}"
        )
    print(f"Report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--text", action="append", default=[])
    args = parser.parse_args()
    report = verify_model(args.model, args.db, args.samples, args.text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
