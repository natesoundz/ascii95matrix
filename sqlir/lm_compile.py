"""Compile the database-derived count model used by :mod:`sqlir.sql_adapter`."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Callable

from .catalog import canonical_json, quote_identifier, rowid_name
SPECIAL_WORDS = ("<unk>", "<s>", "</s>")


def norm_word(value: str) -> str:
    """Normalize one model word, rejecting values outside the model alphabet."""
    result = value.strip().lower()
    if not result or any(ord(char) < 33 or ord(char) > 126 for char in result):
        raise ValueError(f"Invalid model word: {value!r}")
    return result


def _safe_word(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return norm_word(str(value))
    except (TypeError, ValueError):
        return None


def _text_affinity(declared_type: str | None) -> bool:
    declared_type = (declared_type or "").upper()
    return any(token in declared_type for token in ("CHAR", "CLOB", "TEXT"))


def _table_info(con: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return list(con.execute(f"PRAGMA table_info({quote_identifier(table)})"))


def _table_map(con: sqlite3.Connection) -> dict[str, str]:
    return {
        name.lower(): name
        for (name,) in con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _column_map(info: list[sqlite3.Row]) -> dict[str, sqlite3.Row]:
    return {row[1].lower(): row for row in info}


def _order_columns(info: list[sqlite3.Row]) -> list[str]:
    primary = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]]
    return primary or [rowid_name([row[1] for row in info])]


def _order_sql(info: list[sqlite3.Row]) -> str:
    return ", ".join(quote_identifier(column) for column in _order_columns(info))


def _add_missing(missing: list[str], value: str) -> None:
    if value not in missing:
        missing.append(value)


def _add_source(sources: dict[str, dict], name: str, sql: str) -> None:
    sources[name] = {
        "sql": sql,
        "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
    }


def _write_vocab(store, spool: sqlite3.Connection) -> tuple[str, str, int]:
    keys_name = "lm:vocab:keys"
    offsets_name = "lm:vocab:offsets"
    keys = store.writer(keys_name, "uint8")
    offsets = store.writer(offsets_name, "<i8")
    offsets.write_i64(0)
    byte_count = 0
    count = 0
    for word in SPECIAL_WORDS:
        encoded = word.encode("utf-8")
        keys.write(encoded)
        byte_count += len(encoded)
        offsets.write_i64(byte_count)
        count += 1
    for (word,) in spool.execute("SELECT word FROM lm_vocab ORDER BY word"):
        encoded = word.encode("utf-8")
        keys.write(encoded)
        byte_count += len(encoded)
        offsets.write_i64(byte_count)
        count += 1
    keys.finish([byte_count])
    offsets.finish([count + 1])
    return keys_name, offsets_name, count


def _write_i64_array(store, name: str, values) -> str:
    writer = store.writer(name, "<i8")
    count = 0
    for value in values:
        writer.write_i64(value)
        count += 1
    writer.finish([count])
    return name


def _write_numeric_csr(
    store,
    spool: sqlite3.Connection,
    prefix: str,
    vocab_size: int,
    query: str,
    value_dtype: str,
    value_writer: Callable[[Any, Any], bytes],
) -> tuple[dict[str, str], int]:
    indptr_name = f"lm:{prefix}:indptr"
    col_name = f"lm:{prefix}:col"
    value_name = f"lm:{prefix}:value"
    indptr = store.writer(indptr_name, "<i8")
    cols = store.writer(col_name, "<i8")
    values = store.writer(value_name, value_dtype)
    indptr.write_i64(0)
    current_id = 0
    nnz = 0
    for word_id, column, value in spool.execute(query):
        word_id = int(word_id)
        while current_id < word_id:
            indptr.write_i64(nnz)
            current_id += 1
        cols.write_i64(int(column))
        values.write(value_writer(column, value))
        nnz += 1
    while current_id < vocab_size:
        indptr.write_i64(nnz)
        current_id += 1
    indptr.finish([vocab_size + 1])
    cols.finish([nnz])
    values.finish([nnz])
    return {
        "indptr": indptr_name,
        "col": col_name,
        "value": value_name,
    }, nnz


def _write_text_csr(
    store,
    spool: sqlite3.Connection,
    prefix: str,
    vocab_size: int,
    query: str,
    value_builder: Callable[[tuple], Any],
) -> tuple[dict[str, str], int]:
    indptr_name = f"lm:{prefix}:indptr"
    rows_name = f"lm:{prefix}:rows"
    offsets_name = f"lm:{prefix}:offsets"
    indptr = store.writer(indptr_name, "<i8")
    rows = store.writer(rows_name, "uint8")
    offsets = store.writer(offsets_name, "<i8")
    indptr.write_i64(0)
    offsets.write_i64(0)
    current_id = 0
    row_count = 0
    byte_count = 0
    for row in spool.execute(query):
        word_id = int(row[0])
        while current_id < word_id:
            indptr.write_i64(row_count)
            current_id += 1
        encoded = canonical_json(value_builder(tuple(row[1:])))
        rows.write(encoded)
        byte_count += len(encoded)
        offsets.write_i64(byte_count)
        row_count += 1
    while current_id < vocab_size:
        indptr.write_i64(row_count)
        current_id += 1
    indptr.finish([vocab_size + 1])
    rows.finish([byte_count])
    offsets.finish([row_count + 1])
    return {
        "indptr": indptr_name,
        "rows": rows_name,
        "offsets": offsets_name,
    }, row_count


def _empty_text_csr(store, prefix: str, vocab_size: int) -> dict[str, str]:
    return _write_text_csr(
        store,
        sqlite3.connect(":memory:"),
        prefix,
        vocab_size,
        "SELECT 0 WHERE 0",
        lambda row: None,
    )[0]


def _build_vocab(
    con: sqlite3.Connection,
    spool: sqlite3.Connection,
    source_tables: dict[str, str],
    sources: dict[str, dict],
    missing: list[str],
) -> tuple[int, bool, bool]:
    spool.execute("DROP TABLE IF EXISTS lm_vocab")
    spool.execute(
        "CREATE TABLE lm_vocab (word TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    spool.execute("DROP TABLE IF EXISTS lm_occ_tokens")
    spool.execute(
        "CREATE TABLE lm_occ_tokens (ordinal INTEGER PRIMARY KEY, "
        "example_id, token_index, word TEXT)"
    )
    occurrence_token_source = False

    def insert_words(cursor) -> None:
        for (value,) in cursor:
            word = _safe_word(value)
            if word is not None:
                spool.execute("INSERT OR IGNORE INTO lm_vocab(word) VALUES (?)", (word,))

    def simple_source(logical: str, table_key: str, column_key: str) -> None:
        table = source_tables.get(table_key)
        if table is None:
            _add_missing(missing, f"{table_key}.{column_key}")
            return
        columns = _column_map(_table_info(con, table))
        column = columns.get(column_key)
        if column is None:
            _add_missing(missing, f"{table_key}.{column_key}")
            return
        sql = (
            f"SELECT {quote_identifier(column[1])} FROM {quote_identifier(table)} "
            f"WHERE {quote_identifier(column[1])} IS NOT NULL"
        )
        _add_source(sources, logical, sql)
        insert_words(con.execute(sql))

    simple_source("vocab:lexeme.normalized", "lexeme", "normalized")

    occurrence = source_tables.get("occurrence")
    occurrence_info = _table_info(con, occurrence) if occurrence else []
    occurrence_columns = _column_map(occurrence_info)
    sequence_source = False
    occurrence_word_column = occurrence_columns.get("text")
    if occurrence_word_column is None:
        _add_missing(missing, "occurrence.text")
        occurrence_word_column = occurrence_columns.get("lemma")
        occurrence_word_name = "lemma"
    else:
        occurrence_word_name = "text"
    if occurrence is None:
        _add_missing(missing, "occurrence.text")
        _add_missing(missing, "occurrence.lemma")
    elif occurrence_word_column is None:
        _add_missing(missing, "occurrence.lemma")
    else:
        order = _order_sql(occurrence_info)
        sql = (
            f"SELECT {quote_identifier(occurrence_word_column[1])} "
            f"FROM {quote_identifier(occurrence)} "
            f"WHERE {quote_identifier(occurrence_word_column[1])} IS NOT NULL"
        )
        _add_source(sources, f"vocab:occurrence.{occurrence_word_name}", sql)
        insert_words(con.execute(sql))
        example = occurrence_columns.get("example_id")
        token_index = occurrence_columns.get("token_index")
        if example is None:
            _add_missing(missing, "occurrence.example_id")
        if token_index is None:
            _add_missing(missing, "occurrence.token_index")
        sequence_source = example is not None and token_index is not None
        if example is None:
            example_sql = "NULL"
        else:
            example_sql = quote_identifier(example[1])
        if token_index is None:
            token_index_sql = "NULL"
        else:
            token_index_sql = quote_identifier(token_index[1])
        ordering = (
            f"{example_sql}, {token_index_sql}, {order}"
            if sequence_source
            else order
        )
        token_sql = (
            f"SELECT {example_sql}, {token_index_sql}, "
            f"{quote_identifier(occurrence_word_column[1])} "
            f"FROM {quote_identifier(occurrence)} ORDER BY {ordering}"
        )
        _add_source(sources, "unigram:occurrence.tokens", token_sql)
        if sequence_source:
            _add_source(sources, "bigrams:occurrence.tokens", token_sql)
        ordinal = 0
        for example_id, token_id, value in con.execute(token_sql):
            spool.execute(
                "INSERT INTO lm_occ_tokens VALUES (?, ?, ?, ?)",
                (ordinal, example_id, token_id, _safe_word(value)),
            )
            ordinal += 1
        occurrence_token_source = True

    for table_key, column_key in (
        ("collocation", "left_lemma"),
        ("collocation", "right_lemma"),
        ("relation", "target_lemma"),
        ("inflection_form", "normalized"),
    ):
        simple_source(
            f"vocab:{table_key}.{column_key}", table_key, column_key
        )

    spool.execute("DROP TABLE IF EXISTS lm_vocab_ids")
    spool.execute(
        "CREATE TABLE lm_vocab_ids (word TEXT PRIMARY KEY, id INTEGER UNIQUE)"
    )
    for index, (word,) in enumerate(
        spool.execute("SELECT word FROM lm_vocab ORDER BY word"), start=3
    ):
        spool.execute("INSERT INTO lm_vocab_ids(word,id) VALUES (?,?)", (word, index))
    spool.commit()
    count = spool.execute("SELECT COUNT(*) FROM lm_vocab_ids").fetchone()[0] + 3
    return int(count), occurrence_token_source, sequence_source


def _build_unigram_and_bigram(
    store,
    spool: sqlite3.Connection,
    vocab_size: int,
    occurrence_tokens: bool,
    sequence_tokens: bool,
) -> tuple[dict[str, str], dict[str, str], int, int]:
    spool.execute("DROP TABLE IF EXISTS lm_unigram_counts")
    spool.execute(
        "CREATE TABLE lm_unigram_counts (word_id INTEGER PRIMARY KEY, count INTEGER)"
    )
    spool.execute("DROP TABLE IF EXISTS lm_bigram_counts")
    spool.execute(
        "CREATE TABLE lm_bigram_counts (src INTEGER, dst INTEGER, count INTEGER, "
        "PRIMARY KEY(src,dst)) WITHOUT ROWID"
    )
    if occurrence_tokens:
        spool.execute("ALTER TABLE lm_occ_tokens ADD COLUMN word_id INTEGER")
        spool.execute(
            "UPDATE lm_occ_tokens SET word_id = ("
            "SELECT id FROM lm_vocab_ids WHERE word = lm_occ_tokens.word)"
        )
        spool.execute(
            "INSERT INTO lm_unigram_counts "
            "SELECT word_id, COUNT(*) FROM lm_occ_tokens "
            "WHERE word_id IS NOT NULL GROUP BY word_id"
        )
        if sequence_tokens:
            add_sql = (
                "INSERT INTO lm_bigram_counts(src,dst,count) VALUES (?,?,1) "
                "ON CONFLICT(src,dst) DO UPDATE SET count=count+1"
            )
            current_example = object()
            previous = None
            for example_id, _, word_id in spool.execute(
                "SELECT example_id, token_index, word_id FROM lm_occ_tokens "
                "ORDER BY example_id, token_index, ordinal"
            ):
                if example_id != current_example:
                    if previous not in (None, 0):
                        spool.execute(add_sql, (previous, 2))
                    current_example = example_id
                    previous = 1
                current = int(word_id) if word_id is not None else 0
                if previous not in (None, 0) and current != 0:
                    spool.execute(add_sql, (previous, current))
                previous = current
            if previous not in (None, 0):
                spool.execute(add_sql, (previous, 2))
    spool.commit()

    unigram_name = "lm:unigram"
    unigram = store.writer(unigram_name, "<i8")
    for word_id in range(vocab_size):
        row = spool.execute(
            "SELECT count FROM lm_unigram_counts WHERE word_id=?", (word_id,)
        ).fetchone()
        unigram.write_i64(0 if row is None else row[0])
    unigram.finish([vocab_size])

    spool.execute("DROP TABLE IF EXISTS lm_unigram_rank")
    spool.execute(
        "CREATE TABLE lm_unigram_rank (id INTEGER PRIMARY KEY, rank INTEGER)"
    )
    rank = 0
    for word_id, _count in spool.execute(
        "SELECT v.id, COALESCE(c.count,0) FROM lm_vocab_ids v "
        "LEFT JOIN lm_unigram_counts c ON c.word_id=v.id "
        "ORDER BY COALESCE(c.count,0) DESC, v.id"
    ):
        rank += 1
        spool.execute(
            "INSERT INTO lm_unigram_rank VALUES (?,?)", (word_id, rank)
        )
    spool.commit()
    rank_name = "lm:unigram_rank"
    rank_writer = store.writer(rank_name, "<i8")
    for _ in range(3):
        rank_writer.write_i64(0)
    for word_id in range(3, vocab_size):
        rank_value = spool.execute(
            "SELECT rank FROM lm_unigram_rank WHERE id=?", (word_id,)
        ).fetchone()[0]
        rank_writer.write_i64(rank_value)
    rank_writer.finish([vocab_size])

    def i64(_column, value) -> bytes:
        return int(value).to_bytes(8, "little", signed=True)

    def f64(_column, value) -> bytes:
        import struct

        return struct.pack("<d", float(value))

    bigram, bigram_nnz = _write_numeric_csr(
        store,
        spool,
        "bigram",
        vocab_size,
        "SELECT src,dst,count FROM lm_bigram_counts ORDER BY src,dst",
        "<i8",
        i64,
    )
    return (
        bigram,
        {"unigram": unigram_name, "unigram_rank": rank_name},
        int(spool.execute("SELECT COUNT(*) FROM lm_unigram_counts").fetchone()[0]),
        bigram_nnz,
    )


def _build_collocations(
    con: sqlite3.Connection,
    store,
    spool: sqlite3.Connection,
    source_tables: dict[str, str],
    source_columns: dict[str, dict[str, sqlite3.Row]],
    sources: dict[str, dict],
    missing: list[str],
    vocab_size: int,
) -> tuple[dict[str, str], int]:
    table = source_tables.get("collocation")
    required = ("left_lemma", "right_lemma", "frequency")
    if table is None:
        for column in required:
            _add_missing(missing, f"collocation.{column}")
        return _empty_numeric_csr(store, "colloc", vocab_size, "<f8"), 0
    columns = source_columns["collocation"]
    if any(column not in columns for column in required):
        for column in required:
            if column not in columns:
                _add_missing(missing, f"collocation.{column}")
        return _empty_numeric_csr(store, "colloc", vocab_size, "<f8"), 0
    info = _table_info(con, table)
    order = _order_sql(info)
    sql = (
        f"SELECT {quote_identifier(columns['left_lemma'][1])}, "
        f"{quote_identifier(columns['right_lemma'][1])}, "
        f"{quote_identifier(columns['frequency'][1])} "
        f"FROM {quote_identifier(table)} ORDER BY {order}"
    )
    _add_source(sources, "colloc:collocation", sql)
    spool.execute("DROP TABLE IF EXISTS lm_colloc_raw")
    spool.execute(
        "CREATE TABLE lm_colloc_raw (seq INTEGER PRIMARY KEY, left_word TEXT, "
        "right_word TEXT, weight REAL)"
    )
    for seq, (left, right, frequency) in enumerate(con.execute(sql)):
        left_word = _safe_word(left)
        right_word = _safe_word(right)
        if left_word is None or right_word is None:
            continue
        weight = 1.0 if frequency is None else float(frequency)
        if not math.isfinite(weight):
            continue
        spool.execute(
            "INSERT INTO lm_colloc_raw VALUES (?,?,?,?)",
            (seq, left_word, right_word, weight),
        )
    spool.execute("DROP TABLE IF EXISTS lm_colloc_counts")
    spool.execute(
        "CREATE TABLE lm_colloc_counts (src INTEGER, dst INTEGER, weight REAL, "
        "PRIMARY KEY(src,dst)) WITHOUT ROWID"
    )
    spool.execute(
        "INSERT INTO lm_colloc_counts "
        "SELECT l.id, r.id, SUM(raw.weight) FROM lm_colloc_raw raw "
        "JOIN lm_vocab_ids l ON l.word=raw.left_word "
        "JOIN lm_vocab_ids r ON r.word=raw.right_word "
        "GROUP BY l.id,r.id"
    )
    spool.commit()

    def f64(_column, value) -> bytes:
        import struct

        return struct.pack("<d", float(value))

    segments, count = _write_numeric_csr(
        store,
        spool,
        "colloc",
        vocab_size,
        "SELECT src,dst,weight FROM lm_colloc_counts ORDER BY src,dst",
        "<f8",
        f64,
    )
    return segments, count


def _empty_numeric_csr(store, prefix: str, vocab_size: int, dtype: str) -> dict[str, str]:
    return _write_numeric_csr(
        store,
        sqlite3.connect(":memory:"),
        prefix,
        vocab_size,
        "SELECT 0,0,0 WHERE 0",
        dtype,
        lambda _column, _value: b"\0" * 8,
    )[0]


def _build_text_sources(
    con: sqlite3.Connection,
    store,
    spool: sqlite3.Connection,
    source_tables: dict[str, str],
    source_columns: dict[str, dict[str, sqlite3.Row]],
    sources: dict[str, dict],
    missing: list[str],
    vocab_size: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], int, int]:
    pos_segments, lemma_segments, inflection_segments = (
        _empty_text_csr(store, prefix, vocab_size)
        for prefix in ("pos", "lemma_of", "inflections")
    )
    counts = {"pos": 0, "lemma_of": 0, "inflections": 0}

    sense = source_tables.get("sense")
    sense_columns = source_columns.get("sense", {})
    if sense is None or "pos" not in sense_columns or "lexeme_id" not in sense_columns:
        _add_missing(missing, "sense.pos")
        _add_missing(missing, "sense.lexeme_id")
    else:
        lexeme = source_tables.get("lexeme")
        lexeme_columns = source_columns.get("lexeme", {})
        if lexeme is None or "lexeme_id" not in lexeme_columns or "normalized" not in lexeme_columns:
            _add_missing(missing, "lexeme.lexeme_id")
            _add_missing(missing, "lexeme.normalized")
        else:
            info = _table_info(con, sense)
            sql = (
                f"SELECT l.{quote_identifier(lexeme_columns['normalized'][1])}, "
                f"s.{quote_identifier(sense_columns['pos'][1])} "
                f"FROM {quote_identifier(sense)} s JOIN {quote_identifier(lexeme)} l "
                f"ON l.{quote_identifier(lexeme_columns['lexeme_id'][1])}="
                f"s.{quote_identifier(sense_columns['lexeme_id'][1])} "
                f"WHERE s.{quote_identifier(sense_columns['pos'][1])} IS NOT NULL "
                f"ORDER BY {_order_sql(info)}"
            )
            _add_source(sources, "pos:sense.pos", sql)
            spool.execute("DROP TABLE IF EXISTS lm_pos_raw")
            spool.execute(
                "CREATE TABLE lm_pos_raw (seq INTEGER PRIMARY KEY, word TEXT, pos TEXT)"
            )
            for seq, (word, pos) in enumerate(con.execute(sql)):
                word = _safe_word(word)
                if word is not None and pos is not None:
                    spool.execute(
                        "INSERT INTO lm_pos_raw VALUES (?,?,?)",
                        (seq, word, str(pos)),
                    )
            spool.execute("DROP TABLE IF EXISTS lm_pos")
            spool.execute(
                "CREATE TABLE lm_pos (word_id INTEGER, pos TEXT, PRIMARY KEY(word_id,pos)) WITHOUT ROWID"
            )
            spool.execute(
                "INSERT OR IGNORE INTO lm_pos "
                "SELECT v.id,p.pos FROM lm_pos_raw p JOIN lm_vocab_ids v ON v.word=p.word"
            )
            pos_segments, counts["pos"] = _write_text_csr(
                store,
                spool,
                "pos",
                vocab_size,
                "SELECT word_id,pos FROM lm_pos ORDER BY word_id,pos",
                lambda row: row[0],
            )

    inflection = source_tables.get("inflection_form")
    inflection_columns = source_columns.get("inflection_form", {})
    required = ("normalized", "lexeme_id")
    if inflection is None or any(column not in inflection_columns for column in required):
        for column in required:
            _add_missing(missing, f"inflection_form.{column}")
    else:
        lexeme = source_tables.get("lexeme")
        lexeme_columns = source_columns.get("lexeme", {})
        if lexeme is None or "lexeme_id" not in lexeme_columns or "normalized" not in lexeme_columns:
            _add_missing(missing, "lexeme.lexeme_id")
            _add_missing(missing, "lexeme.normalized")
        else:
            info = _table_info(con, inflection)
            sql = (
                f"SELECT i.{quote_identifier(inflection_columns['normalized'][1])}, "
                f"l.{quote_identifier(lexeme_columns['normalized'][1])} "
                f"FROM {quote_identifier(inflection)} i JOIN {quote_identifier(lexeme)} l "
                f"ON l.{quote_identifier(lexeme_columns['lexeme_id'][1])}="
                f"i.{quote_identifier(inflection_columns['lexeme_id'][1])} "
                f"ORDER BY {_order_sql(info)}"
            )
            _add_source(sources, "inflections:inflection_form", sql)
            spool.execute("DROP TABLE IF EXISTS lm_inflection_raw")
            spool.execute(
                "CREATE TABLE lm_inflection_raw (seq INTEGER PRIMARY KEY, form TEXT, base TEXT)"
            )
            for seq, (form, base) in enumerate(con.execute(sql)):
                form = _safe_word(form)
                base = _safe_word(base)
                if form is not None and base is not None:
                    spool.execute(
                        "INSERT INTO lm_inflection_raw VALUES (?,?,?)",
                        (seq, form, base),
                    )
            spool.execute("DROP TABLE IF EXISTS lm_lemma_of")
            spool.execute(
                "CREATE TABLE lm_lemma_of (form_id INTEGER, base TEXT, "
                "PRIMARY KEY(form_id,base)) WITHOUT ROWID"
            )
            spool.execute(
                "INSERT OR IGNORE INTO lm_lemma_of "
                "SELECT f.id,r.base FROM lm_inflection_raw r "
                "JOIN lm_vocab_ids f ON f.word=r.form "
                "JOIN lm_vocab_ids b ON b.word=r.base"
            )
            spool.execute("DROP TABLE IF EXISTS lm_inflections")
            spool.execute(
                "CREATE TABLE lm_inflections (base_id INTEGER, form TEXT, "
                "PRIMARY KEY(base_id,form)) WITHOUT ROWID"
            )
            spool.execute(
                "INSERT OR IGNORE INTO lm_inflections "
                "SELECT b.id,r.form FROM lm_inflection_raw r "
                "JOIN lm_vocab_ids f ON f.word=r.form "
                "JOIN lm_vocab_ids b ON b.word=r.base"
            )
            lemma_segments, counts["lemma_of"] = _write_text_csr(
                store,
                spool,
                "lemma_of",
                vocab_size,
                "SELECT form_id,base FROM lm_lemma_of ORDER BY form_id,base",
                lambda row: row[0],
            )
            inflection_segments, counts["inflections"] = _write_text_csr(
                store,
                spool,
                "inflections",
                vocab_size,
                "SELECT base_id,form FROM lm_inflections ORDER BY base_id,form",
                lambda row: row[0],
            )
    return (
        pos_segments,
        lemma_segments,
        inflection_segments,
        counts["pos"],
        counts["lemma_of"] + counts["inflections"],
    )


def _choose_definition_source(
    con: sqlite3.Connection,
    source_tables: dict[str, str],
    source_columns: dict[str, dict[str, sqlite3.Row]],
    override: str | None,
    missing: list[str],
) -> tuple[str, str, str] | None:
    if override:
        if "." not in override:
            raise ValueError("--define-source must be TABLE.COLUMN")
        table_key, column_key = override.split(".", 1)
        table = source_tables.get(table_key.lower())
        column = source_columns.get(table_key.lower(), {}).get(column_key.lower())
        if table is None or column is None or not _text_affinity(column[2]):
            raise ValueError(f"Invalid definition source: {override}")
        return table, column[1], f"{table}.{column[1]}"
    for table_key in ("sense", "webster_entry", "definition", "entry"):
        table = source_tables.get(table_key)
        if table is None:
            continue
        for column_key in (
            "definition",
            "definition_text",
            "gloss",
            "def",
            "meaning",
            "text",
            "body",
        ):
            column = source_columns[table_key].get(column_key)
            if column is not None and _text_affinity(column[2]):
                return table, column[1], f"{table}.{column[1]}"
    _add_missing(missing, "definition source column")
    return None


def _build_definitions(
    con: sqlite3.Connection,
    store,
    spool: sqlite3.Connection,
    source_tables: dict[str, str],
    source_columns: dict[str, dict[str, sqlite3.Row]],
    sources: dict[str, dict],
    missing: list[str],
    vocab_size: int,
    override: str | None,
) -> tuple[dict[str, str], tuple[str, str] | None, int]:
    selected = _choose_definition_source(
        con, source_tables, source_columns, override, missing
    )
    if selected is None:
        return _empty_text_csr(store, "definitions", vocab_size), None, 0
    table, definition_column, source_label = selected
    columns = source_columns[table.lower()]
    link_column = columns.get("lexeme_id")
    link_table = source_tables.get("lexeme")
    if link_column is not None:
        lexeme_columns = source_columns.get("lexeme", {})
        if (
            link_table is None
            or "lexeme_id" not in lexeme_columns
            or "normalized" not in lexeme_columns
        ):
            _add_missing(missing, "lexeme.lexeme_id")
            _add_missing(missing, "lexeme.normalized")
            return _empty_text_csr(store, "definitions", vocab_size), None, 0
        link_expression = f"l.{quote_identifier(lexeme_columns['normalized'][1])}"
        from_sql = (
            f"FROM {quote_identifier(table)} d JOIN {quote_identifier(link_table)} l "
            f"ON d.{quote_identifier(link_column[1])}=l."
            f"{quote_identifier(lexeme_columns['lexeme_id'][1])}"
        )
    else:
        own_link = next(
            (columns.get(name) for name in ("normalized", "headword", "word") if columns.get(name)),
            None,
        )
        if own_link is None:
            _add_missing(missing, f"{source_label} link column")
            return _empty_text_csr(store, "definitions", vocab_size), None, 0
        link_expression = f"d.{quote_identifier(own_link[1])}"
        from_sql = f"FROM {quote_identifier(table)} d"
    info = _table_info(con, table)
    sql = (
        f"SELECT {link_expression}, d.{quote_identifier(definition_column)} "
        f"{from_sql} "
        f"WHERE d.{quote_identifier(definition_column)} IS NOT NULL "
        f"ORDER BY " + ", ".join(f"d.{quote_identifier(column)}" for column in _order_columns(info))
    )
    _add_source(sources, "definitions", sql)
    spool.execute("DROP TABLE IF EXISTS lm_def_raw")
    spool.execute(
        "CREATE TABLE lm_def_raw (seq INTEGER PRIMARY KEY, word TEXT, text TEXT)"
    )
    for seq, (word, text) in enumerate(con.execute(sql)):
        word = _safe_word(word)
        if word is not None and text is not None:
            spool.execute("INSERT INTO lm_def_raw VALUES (?,?,?)", (seq, word, str(text)))
    spool.execute("DROP TABLE IF EXISTS lm_definitions")
    spool.execute(
        "CREATE TABLE lm_definitions (word_id INTEGER, text TEXT, first_order INTEGER, "
        "PRIMARY KEY(word_id,text)) WITHOUT ROWID"
    )
    spool.execute(
        "INSERT OR IGNORE INTO lm_definitions "
        "SELECT v.id,r.text,r.seq FROM lm_def_raw r JOIN lm_vocab_ids v ON v.word=r.word "
        "ORDER BY r.seq"
    )
    segments, count = _write_text_csr(
        store,
        spool,
        "definitions",
        vocab_size,
        "SELECT word_id,text FROM lm_definitions ORDER BY word_id,first_order",
        lambda row: row[0],
    )
    return segments, (table, definition_column), count


def _build_relations(
    con: sqlite3.Connection,
    store,
    spool: sqlite3.Connection,
    source_tables: dict[str, str],
    source_columns: dict[str, dict[str, sqlite3.Row]],
    sources: dict[str, dict],
    missing: list[str],
    vocab_size: int,
) -> tuple[dict[str, str], int]:
    table = source_tables.get("relation")
    columns = source_columns.get("relation", {})
    required = ("source_lexeme_id", "target_lemma")
    if table is None or any(column not in columns for column in required):
        for column in required:
            if table is None or column not in columns:
                _add_missing(missing, f"relation.{column}")
        return _empty_text_csr(store, "relations", vocab_size), 0
    lexeme = source_tables.get("lexeme")
    lexeme_columns = source_columns.get("lexeme", {})
    if (
        lexeme is None
        or "lexeme_id" not in lexeme_columns
        or "normalized" not in lexeme_columns
    ):
        _add_missing(missing, "lexeme.lexeme_id")
        _add_missing(missing, "lexeme.normalized")
        return _empty_text_csr(store, "relations", vocab_size), 0
    info = _table_info(con, table)
    relation_type = columns.get("relation_type")
    type_sql = (
        f"r.{quote_identifier(relation_type[1])}" if relation_type else "NULL"
    )
    sql = (
        f"SELECT l.{quote_identifier(lexeme_columns['normalized'][1])}, {type_sql}, "
        f"r.{quote_identifier(columns['target_lemma'][1])} "
        f"FROM {quote_identifier(table)} r JOIN {quote_identifier(lexeme)} l "
        f"ON l.{quote_identifier(lexeme_columns['lexeme_id'][1])}="
        f"r.{quote_identifier(columns['source_lexeme_id'][1])} "
        f"WHERE r.{quote_identifier(columns['target_lemma'][1])} IS NOT NULL "
        f"ORDER BY " + ", ".join(f"r.{quote_identifier(column)}" for column in _order_columns(info))
    )
    _add_source(sources, "relations", sql)
    spool.execute("DROP TABLE IF EXISTS lm_relations_raw")
    spool.execute(
        "CREATE TABLE lm_relations_raw (seq INTEGER PRIMARY KEY, word TEXT, "
        "relation_type TEXT, target TEXT)"
    )
    for seq, (word, relation_type_value, target) in enumerate(con.execute(sql)):
        word = _safe_word(word)
        if word is None or target is None:
            continue
        target = _safe_word(target)
        if target is None:
            continue
        type_value = "\x00" if relation_type_value is None else str(relation_type_value)
        spool.execute(
            "INSERT INTO lm_relations_raw VALUES (?,?,?,?)",
            (seq, word, type_value, target),
        )
    spool.execute("DROP TABLE IF EXISTS lm_relations")
    spool.execute(
        "CREATE TABLE lm_relations (word_id INTEGER, relation_type TEXT, target TEXT, "
        "PRIMARY KEY(word_id,relation_type,target)) WITHOUT ROWID"
    )
    spool.execute(
        "INSERT OR IGNORE INTO lm_relations "
        "SELECT v.id,r.relation_type,r.target FROM lm_relations_raw r "
        "JOIN lm_vocab_ids v ON v.word=r.word"
    )
    segments, count = _write_text_csr(
        store,
        spool,
        "relations",
        vocab_size,
        "SELECT word_id,relation_type,target FROM lm_relations "
        "ORDER BY word_id,relation_type,target",
        lambda row: [None if row[0] == "\x00" else row[0], row[1]],
    )
    return segments, count


def compile_lm(
    con: sqlite3.Connection,
    workbook: Path,
    store,
    spool: sqlite3.Connection,
    define_source: str | None = None,
) -> dict:
    """Compile the count model and all automatic-scan indexes into ``store``."""
    source_tables = _table_map(con)
    source_columns = {
        key: _column_map(_table_info(con, table))
        for key, table in source_tables.items()
    }
    sources: dict[str, dict] = {}
    missing: list[str] = []
    vocab_size, occurrence_tokens, sequence_tokens = _build_vocab(
        con, spool, source_tables, sources, missing
    )
    vocab_keys, vocab_offsets, actual_vocab_size = _write_vocab(store, spool)
    if actual_vocab_size != vocab_size:
        raise RuntimeError("Vocabulary spool count changed while writing segments")
    vocab_segments = {"keys": vocab_keys, "offsets": vocab_offsets}

    bigram_segments, count_segments, unigram_count, bigram_count = (
        _build_unigram_and_bigram(
            store, spool, vocab_size, occurrence_tokens, sequence_tokens
        )
    )
    colloc_segments, colloc_count = _build_collocations(
        con,
        store,
        spool,
        source_tables,
        source_columns,
        sources,
        missing,
        vocab_size,
    )
    pos_segments, lemma_segments, inflection_segments, pos_count, lemma_count = (
        _build_text_sources(
            con,
            store,
            spool,
            source_tables,
            source_columns,
            sources,
            missing,
            vocab_size,
        )
    )
    definition_segments, definition_source, definition_count = _build_definitions(
        con,
        store,
        spool,
        source_tables,
        source_columns,
        sources,
        missing,
        vocab_size,
        define_source,
    )
    relation_segments, relation_count = _build_relations(
        con,
        store,
        spool,
        source_tables,
        source_columns,
        sources,
        missing,
        vocab_size,
    )

    workbook_name = "lm:workbook:source"
    store.add_bytes(workbook_name, workbook.read_bytes())
    return {
        "vocab_size": vocab_size,
        "sources": sources,
        "sources_missing": missing,
        "definition_source": (
            {"table": definition_source[0], "column": definition_source[1]}
            if definition_source
            else None
        ),
        "workbook_segment": workbook_name,
        "segments": {
            "vocab": vocab_segments,
            "unigram": count_segments["unigram"],
            "unigram_rank": count_segments["unigram_rank"],
            "bigram": bigram_segments,
            "colloc": colloc_segments,
            "pos": pos_segments,
            "lemma_of": lemma_segments,
            "inflections": inflection_segments,
            "definitions": definition_segments,
            "relations": relation_segments,
        },
        "counts": {
            "vocab": vocab_size,
            "unigram_nonzero": unigram_count,
            "bigram_pairs": bigram_count,
            "colloc_pairs": colloc_count,
            "pos_values": pos_count,
            "lemma_values": lemma_count,
            "definitions": definition_count,
            "relations": relation_count,
        },
    }
