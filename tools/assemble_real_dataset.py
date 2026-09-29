#!/usr/bin/env python3
"""Assemble a provenance-preserving real ASCII95 dataset.

Inputs:
  * webster_full_lexicon.sqlite (optional but strongly recommended)
  * one or more local corpus files/directories (optional)

Outputs in --out:
  * records.jsonl          real text/code records with occurrence annotations
  * lexical_snapshot.jsonl full lexical facts once per observed normalized lexeme
  * rejections.jsonl       records excluded without silent normalization
  * manifest.json          hashes, counts, source inventory, invariants

This file assembles evidence. It does NOT invent 437-D measurements. The next
binding stage maps these preserved annotations to exact canonical registry IDs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sqlite3
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

ASCII_MIN = 32
ASCII_MAX = 126
TEXT_OBJECTIVE = "TEXT_CAUSAL"
CODE_OBJECTIVE = "CODE_CAUSAL"
DEFAULT_DB = Path.home() / "Downloads" / "webster_full_lexicon.sqlite"
DEFAULT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".yaml", ".yml"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(x) for x in parts).encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()[:24]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().replace("_", " ")).casefold()


def ascii95(text: str) -> bool:
    return bool(text) and all(ASCII_MIN <= ord(ch) <= ASCII_MAX for ch in text)


def char_roles(ch: str) -> list[str]:
    roles: list[str] = []
    if ch == " ": roles.append("SPACE")
    if ch.isalpha(): roles.append("LETTER")
    if ch.isupper(): roles.append("UPPERCASE")
    if ch.islower(): roles.append("LOWERCASE")
    if ch.isdigit(): roles.append("DIGIT")
    if ch == "_": roles.append("UNDERSCORE")
    if ch in "'\"\`": roles.append("QUOTE")
    if ch in "()[]{}": roles.append("BRACKET")
    if ch in "+-*/%=<>!&|^~:@": roles.append("OPERATOR_OR_DELIMITER")
    if ch in ".,;?": roles.append("PUNCTUATION")
    if not roles: roles.append("OTHER_PRINTABLE")
    return roles


def role_vector(text: str) -> list[list[str]]:
    return [char_roles(ch) for ch in text]


def connect_ro(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con


def tables(con: sqlite3.Connection) -> set[str]:
    return {
        str(r[0])
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"
        )
    }


def table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in con.execute(f'PRAGMA table_info("{table}")')}


def rows_as_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [{k: r[k] for k in r.keys()} for r in rows]


def align_tokens(sentence: str, tokens: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach character spans without changing token or sentence strings."""
    cursor = 0
    aligned: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for token in sorted(tokens, key=lambda x: int(x.get("token_index", 0))):
        token_text = str(token.get("text", ""))
        start = sentence.find(token_text, cursor)
        if start < 0:
            start = sentence.find(token_text, max(0, cursor - 2))
        item = dict(token)
        if start < 0:
            item["char_start"] = None
            item["char_end"] = None
            failures.append({
                "token_index": token.get("token_index"),
                "text": token_text,
                "cursor": cursor,
            })
        else:
            end = start + len(token_text)
            item["char_start"] = start
            item["char_end"] = end
            cursor = end
        aligned.append(item)
    return aligned, failures


def fetch_source(con: sqlite3.Connection, source_id: int | None) -> dict[str, Any]:
    if source_id is None or "source" not in tables(con):
        return {}
    row = con.execute("SELECT * FROM source WHERE source_id=?", (source_id,)).fetchone()
    return dict(row) if row else {}


def fetch_structure(
    con: sqlite3.Connection,
    source_id: int,
    source_record: str,
    sentence: str,
    origin: str,
) -> dict[str, Any]:
    present = tables(con)
    if "sentence_instance" not in present:
        return {"nodes": [], "edges": [], "observations": [], "discourse": []}
    sr = con.execute(
        """SELECT sentence_id FROM sentence_instance
           WHERE source_id=? AND source_record=? AND sentence=? AND origin=?
           ORDER BY sentence_id LIMIT 1""",
        (source_id, source_record, sentence, origin),
    ).fetchone()
    if not sr:
        return {"nodes": [], "edges": [], "observations": [], "discourse": []}
    sentence_id = int(sr[0])
    out: dict[str, Any] = {"sentence_id": sentence_id}
    for key, table in (
        ("nodes", "structure_node"),
        ("edges", "structure_edge"),
        ("observations", "structural_observation"),
        ("discourse", "discourse_annotation"),
    ):
        out[key] = (
            rows_as_dicts(
                con.execute(
                    f'SELECT * FROM "{table}" WHERE sentence_id=? ORDER BY 1',
                    (sentence_id,),
                )
            )
            if table in present
            else []
        )
    return out


def fetch_example_tokens(con: sqlite3.Connection, example_id: int) -> list[dict[str, Any]]:
    present = tables(con)
    if "occurrence" not in present:
        return []
    result: list[dict[str, Any]] = []
    for row in con.execute(
        "SELECT * FROM occurrence WHERE example_id=? ORDER BY token_index",
        (example_id,),
    ):
        item = dict(row)
        occurrence_id = int(item["occurrence_id"])
        item["classifications"] = (
            rows_as_dicts(
                con.execute(
                    """SELECT category,subcategory,rule_id,warranted,details_json
                       FROM occurrence_classification
                       WHERE occurrence_id=?
                       ORDER BY category,subcategory,rule_id""",
                    (occurrence_id,),
                )
            )
            if "occurrence_classification" in present
            else []
        )
        result.append(item)
    return result


def observed_lexemes_from_tokens(tokens: Iterable[dict[str, Any]]) -> set[str]:
    result = set()
    for token in tokens:
        lemma = str(token.get("lemma", "") or token.get("text", ""))
        normalized = norm(lemma)
        if normalized:
            result.add(normalized)
    return result


def lexical_snapshot(con: sqlite3.Connection, normalized: str) -> dict[str, Any]:
    present = tables(con)
    if "lexeme" not in present:
        return {"normalized": normalized, "lexeme": None, "facts": {}}
    lex = con.execute(
        "SELECT * FROM lexeme WHERE normalized=?",
        (normalized,),
    ).fetchone()
    if not lex:
        return {"normalized": normalized, "lexeme": None, "facts": {}}

    lexeme = dict(lex)
    lexeme_id = int(lexeme["lexeme_id"])
    facts: dict[str, Any] = {}
    simple = {
        "forms": ("form", "lexeme_id"),
        "senses": ("sense", "lexeme_id"),
        "evidence": ("evidence", "lexeme_id"),
        "pronunciation": ("pronunciation", "lexeme_id"),
        "inflections": ("inflection_form", "lexeme_id"),
        "domain_assignments": ("domain_assignment", "lexeme_id"),
        "register_frequency": ("register_frequency", "lexeme_id"),
        "primitive_assignments": ("primitive_assignment", "lexeme_id"),
    }
    for key, (table, column) in simple.items():
        if table in present and column in table_columns(con, table):
            facts[key] = rows_as_dicts(
                con.execute(
                    f'SELECT * FROM "{table}" WHERE "{column}"=? ORDER BY 1',
                    (lexeme_id,),
                )
            )

    if "relation" in present:
        facts["relations"] = rows_as_dicts(
            con.execute(
                """SELECT * FROM relation
                   WHERE source_lexeme_id=?
                   ORDER BY relation_type,target_lemma,target_sense_key""",
                (lexeme_id,),
            )
        )

    lemma = str(lexeme.get("lemma", ""))
    if "collocation" in present:
        facts["collocations_left"] = rows_as_dicts(
            con.execute(
                """SELECT * FROM collocation
                   WHERE lower(left_lemma)=lower(?)
                   ORDER BY relation_type,distance,frequency DESC""",
                (lemma,),
            )
        )
        facts["collocations_right"] = rows_as_dicts(
            con.execute(
                """SELECT * FROM collocation
                   WHERE lower(right_lemma)=lower(?)
                   ORDER BY relation_type,distance,frequency DESC""",
                (lemma,),
            )
        )

    source_ids = set()
    for fact_rows in facts.values():
        if isinstance(fact_rows, list):
            for row in fact_rows:
                if isinstance(row, dict) and row.get("source_id") is not None:
                    source_ids.add(int(row["source_id"]))
    sources = []
    if "source" in present:
        for source_id in sorted(source_ids):
            row = con.execute(
                "SELECT * FROM source WHERE source_id=?",
                (source_id,),
            ).fetchone()
            if row:
                sources.append(dict(row))

    return {
        "normalized": normalized,
        "lexeme": lexeme,
        "facts": facts,
        "sources": sources,
    }


def iter_db_records(
    con: sqlite3.Connection,
    origins: set[str],
) -> Iterator[tuple[dict[str, Any], set[str], list[dict[str, Any]]]]:
    if "example_sentence" not in tables(con):
        return
    seen: set[tuple[int, str, str, str]] = set()
    rows = con.execute(
        """SELECT example_id,source_id,sentence,origin,source_record
           FROM example_sentence
           ORDER BY example_id"""
    )
    for row in rows:
        origin = str(row["origin"])
        if origins and origin not in origins:
            continue
        source_id = int(row["source_id"] or 0)
        source_record = str(row["source_record"])
        sentence = str(row["sentence"])
        key = (source_id, source_record, sentence, origin)
        if key in seen:
            continue
        seen.add(key)

        tokens = fetch_example_tokens(con, int(row["example_id"]))
        aligned, failures = align_tokens(sentence, tokens)
        record = {
            "id": "db-" + stable_id(source_id, source_record, origin, sentence),
            "dataset": "webster_full_lexicon",
            "objective": TEXT_OBJECTIVE,
            "text": sentence,
            "ground_truth_positions": list(range(len(sentence))),
            "roles": role_vector(sentence),
            "provenance": {
                "source_kind": "lexicon_database_sentence",
                "database_source_id": source_id,
                "source_record": source_record,
                "origin": origin,
                "source": fetch_source(con, source_id),
            },
            "annotations": {
                "tokens": aligned,
                "token_alignment_failures": failures,
                "structure": fetch_structure(
                    con,
                    source_id,
                    source_record,
                    sentence,
                    origin,
                ),
            },
        }
        yield record, observed_lexemes_from_tokens(aligned), failures


def tokenize_python(
    path: Path,
    text: str,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    tokens_by_line: dict[int, list[dict[str, Any]]] = defaultdict(list)
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            sl, sc = token.start
            el, ec = token.end
            tokens_by_line[sl].append(
                {
                    "type": tokenize.tok_name.get(token.type, str(token.type)),
                    "text": token.string,
                    "start_line": sl,
                    "start_col": sc,
                    "end_line": el,
                    "end_col": ec,
                }
            )
    except (tokenize.TokenError, IndentationError):
        pass

    nodes_by_line: dict[int, list[dict[str, Any]]] = defaultdict(list)
    try:
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not hasattr(node, "lineno"):
                continue
            start_line = int(getattr(node, "lineno"))
            end_line = int(getattr(node, "end_lineno", start_line) or start_line)
            item = {
                "node": type(node).__name__,
                "start_line": start_line,
                "start_col": int(getattr(node, "col_offset", 0) or 0),
                "end_line": end_line,
                "end_col": int(getattr(node, "end_col_offset", 0) or 0),
            }
            for line in range(start_line, end_line + 1):
                nodes_by_line[line].append(item)
    except (SyntaxError, ValueError):
        pass
    return tokens_by_line, nodes_by_line


def iter_corpus_files(paths: Iterable[Path], suffixes: set[str]) -> Iterator[Path]:
    seen = set()
    for path in paths:
        path = path.resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = [
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.casefold() in suffixes
            ]
        else:
            continue
        for candidate in sorted(candidates):
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def iter_file_records(path: Path) -> Iterator[tuple[dict[str, Any], set[str]]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    file_sha = sha256_file(path)
    py_tokens: dict[int, list[dict[str, Any]]] = {}
    py_nodes: dict[int, list[dict[str, Any]]] = {}
    if path.suffix.casefold() == ".py":
        py_tokens, py_nodes = tokenize_python(path, raw)

    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            continue
        annotations: dict[str, Any] = {"line_number": line_number}
        if path.suffix.casefold() == ".py":
            annotations["python_tokens"] = py_tokens.get(line_number, [])
            annotations["python_ast_nodes"] = py_nodes.get(line_number, [])

        record = {
            "id": "file-" + stable_id(file_sha, line_number, line),
            "dataset": "local_corpus",
            "objective": (
                CODE_OBJECTIVE
                if path.suffix.casefold() == ".py"
                else TEXT_OBJECTIVE
            ),
            "text": line,
            "ground_truth_positions": list(range(len(line))),
            "roles": role_vector(line),
            "provenance": {
                "source_kind": "local_file_line",
                "path": str(path),
                "sha256": file_sha,
                "line_number": line_number,
            },
            "annotations": annotations,
        }
        lexemes = {
            norm(match.group(0))
            for match in re.finditer(r"[A-Za-z][A-Za-z'_-]*", line)
            if norm(match.group(0))
        }
        yield record, lexemes


def write_jsonl_line(handle, obj: dict[str, Any]) -> None:
    handle.write(
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )


def global_snapshot_rows(con: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    """Stream non-lexeme-global evidence without duplicating it per occurrence."""
    present = tables(con)
    selected = (
        "source",
        "structural_rule",
        "controlled_term",
        "violation_pattern",
        "construction_frequency",
        "document_instance",
        "document_section",
    )
    for table in selected:
        if table not in present:
            continue
        for row in con.execute(f'SELECT * FROM "{table}" ORDER BY 1'):
            yield {"table": table, "row": dict(row)}

    if "unicode_character" in present:
        for row in con.execute(
            """SELECT * FROM unicode_character
               WHERE codepoint BETWEEN ? AND ?
               ORDER BY codepoint""",
            (ASCII_MIN, ASCII_MAX),
        ):
            yield {"table": "unicode_character", "row": dict(row)}

    # Preserve correction/violation evidence separately from positive examples.
    if "error_occurrence" in present:
        for row in con.execute('SELECT * FROM "error_occurrence" ORDER BY error_id'):
            yield {"table": "error_occurrence", "row": dict(row)}


def database_table_counts(con: sqlite3.Connection) -> dict[str, int]:
    result: dict[str, int] = {}
    for table in sorted(tables(con)):
        try:
            result[table] = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        except sqlite3.DatabaseError:
            result[table] = -1
    return result


def assemble(
    db_path: Path | None,
    corpus_paths: list[Path],
    out: Path,
    origins: set[str],
    suffixes: set[str],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    lexical_path = out / "lexical_snapshot.jsonl"
    global_path = out / "global_snapshot.jsonl"
    rejects_path = out / "rejections.jsonl"

    stats = Counter()
    objective_counts = Counter()
    source_counts = Counter()
    observed_lexemes: set[str] = set()
    db_sha = None
    db_table_counts: dict[str, int] = {}
    con: sqlite3.Connection | None = None

    with records_path.open("w", encoding="utf-8") as records_file, rejects_path.open(
        "w",
        encoding="utf-8",
    ) as rejects_file:
        if db_path is not None:
            if not db_path.exists():
                raise FileNotFoundError(db_path)
            db_sha = sha256_file(db_path)
            con = connect_ro(db_path)
            db_table_counts = database_table_counts(con)
            for record, lexemes, alignment_failures in iter_db_records(con, origins):
                if not ascii95(record["text"]):
                    write_jsonl_line(
                        rejects_file,
                        {"reason": "NON_ASCII95_OR_EMPTY", "record": record},
                    )
                    stats["rejected"] += 1
                    continue
                write_jsonl_line(records_file, record)
                observed_lexemes.update(lexemes)
                stats["records"] += 1
                stats["characters"] += len(record["text"])
                stats["ground_truth_positions"] += len(
                    record["ground_truth_positions"]
                )
                stats["token_alignment_failures"] += len(alignment_failures)
                objective_counts[record["objective"]] += 1
                source_counts["lexicon_database_sentence"] += 1

        for path in iter_corpus_files(corpus_paths, suffixes):
            try:
                for record, lexemes in iter_file_records(path):
                    if not ascii95(record["text"]):
                        write_jsonl_line(
                            rejects_file,
                            {"reason": "NON_ASCII95_OR_EMPTY", "record": record},
                        )
                        stats["rejected"] += 1
                        continue
                    write_jsonl_line(records_file, record)
                    observed_lexemes.update(lexemes)
                    stats["records"] += 1
                    stats["characters"] += len(record["text"])
                    stats["ground_truth_positions"] += len(
                        record["ground_truth_positions"]
                    )
                    objective_counts[record["objective"]] += 1
                    source_counts["local_file_line"] += 1
            except (OSError, UnicodeError) as exc:
                write_jsonl_line(
                    rejects_file,
                    {
                        "reason": "FILE_READ_ERROR",
                        "path": str(path),
                        "error": str(exc),
                    },
                )
                stats["rejected_files"] += 1

    lexical_written = 0
    global_written = 0
    with lexical_path.open("w", encoding="utf-8") as lexical_file:
        if con is not None:
            for normalized in sorted(observed_lexemes):
                write_jsonl_line(
                    lexical_file,
                    lexical_snapshot(con, normalized),
                )
                lexical_written += 1

    with global_path.open("w", encoding="utf-8") as global_file:
        if con is not None:
            for item in global_snapshot_rows(con):
                write_jsonl_line(global_file, item)
                global_written += 1

    if con is not None:
        con.close()

    manifest = {
        "format": "ascii95-real-evidence-arena-v1",
        "invariants": {
            "text_normalized_silently": False,
            "non_ascii95_records_rejected": True,
            "ground_truth_default": "every retained character occurrence",
            "annotations_are_evidence_not_embedding_coordinates": True,
            "registry_binding_required_next": True,
        },
        "database": (
            None
            if db_path is None
            else {"path": str(db_path.resolve()), "sha256": db_sha}
        ),
        "corpus_roots": [str(path.resolve()) for path in corpus_paths],
        "origins": sorted(origins),
        "suffixes": sorted(suffixes),
        "counts": dict(stats),
        "objective_counts": dict(objective_counts),
        "source_counts": dict(source_counts),
        "observed_lexemes": len(observed_lexemes),
        "lexical_snapshots": lexical_written,
        "global_snapshot_rows": global_written,
        "database_table_counts": db_table_counts,
        "outputs": {
            "records": {
                "path": records_path.name,
                "sha256": sha256_file(records_path),
            },
            "lexical_snapshot": {
                "path": lexical_path.name,
                "sha256": sha256_file(lexical_path),
            },
            "global_snapshot": {
                "path": global_path.name,
                "sha256": sha256_file(global_path),
            },
            "rejections": {
                "path": rejects_path.name,
                "sha256": sha256_file(rejects_path),
            },
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--no-lexicon-db", action="store_true")
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--origins",
        default="real_ud,real_wordnet,user_supplied",
        help="comma-separated example_sentence origins to include",
    )
    parser.add_argument(
        "--suffixes",
        default=",".join(sorted(DEFAULT_SUFFIXES)),
        help="comma-separated local corpus suffixes",
    )
    args = parser.parse_args()

    db_path = None if args.no_lexicon_db else args.lexicon_db
    origins = {
        item.strip()
        for item in args.origins.split(",")
        if item.strip()
    }
    suffixes = {
        item.strip().casefold()
        for item in args.suffixes.split(",")
        if item.strip()
    }
    manifest = assemble(
        db_path,
        args.corpus,
        args.out,
        origins,
        suffixes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
