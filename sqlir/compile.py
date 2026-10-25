"""Compile a read-only SQLite database into a frozen, memory-mappable model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .catalog import (
    QuerySpec,
    build_catalog,
    canonical_json,
    canonical_key,
    canonical_row,
    quote_identifier,
    rowid_name,
)
from .conical_sql_ir_transformer import (
    QUERIES,
    compile_channel,
    direct_operator,
    load_workbook,
)


CHUNK_SIZE = 8 * 1024 * 1024


def status(message: str, eta: str = "under 5s") -> None:
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class MemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                ] + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ]

            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(MemoryCounters),
                wintypes.DWORD,
            ]
            if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
                raise OSError("GetProcessMemoryInfo failed")
            rss = counters.WorkingSetSize / 2**20
        else:
            rss = (
                int(Path("/proc/self/statm").read_text().split()[1])
                * os.sysconf("SC_PAGE_SIZE")
                / 2**20
            )
        ram = f"{rss:.1f} MiB"
    except (OSError, ValueError, AttributeError):
        ram = "unavailable on this platform"
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message} | "
        f"RAM={ram} | VRAM=0 (CPU) | ETA={eta}",
        flush=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


class SegmentStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.descriptors: dict[str, dict] = {}
        self.paths: dict[str, Path] = {}

    def writer(self, name: str, dtype: str) -> "SegmentWriter":
        safe_name = hashlib.sha256(name.encode("utf-8")).hexdigest()
        path = self.directory / f"{safe_name}.segment"
        self.paths[name] = path
        return SegmentWriter(self, name, dtype, path)

    def add_bytes(self, name: str, value: bytes) -> None:
        writer = self.writer(name, "uint8")
        writer.write(value)
        writer.finish([len(value)])

    def add_array(self, name: str, value: np.ndarray, dtype: str) -> None:
        array = np.asarray(value, dtype=np.dtype(dtype), order="C")
        writer = self.writer(name, dtype)
        writer.write(array.tobytes(order="C"))
        writer.finish(list(array.shape))


class SegmentWriter:
    def __init__(self, store: SegmentStore, name: str, dtype: str, path: Path):
        self.store = store
        self.name = name
        self.dtype = dtype
        self.path = path
        self.handle: BinaryIO = path.open("wb")
        self.nbytes = 0

    def write(self, value: bytes) -> None:
        self.handle.write(value)
        self.nbytes += len(value)

    def write_i64(self, value: int) -> None:
        self.write(struct.pack("<q", int(value)))

    def finish(self, shape: list[int]) -> None:
        self.handle.close()
        self.store.descriptors[self.name] = {
            "dtype": self.dtype,
            "shape": shape,
            "nbytes": self.nbytes,
        }


def _database_tables(con: sqlite3.Connection) -> list[tuple[str, str]]:
    return list(
        con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    )


def _table_schema(con: sqlite3.Connection, table: str, create_sql: str) -> dict:
    columns = list(con.execute(f"PRAGMA table_info({quote_identifier(table)})"))
    without_rowid = "WITHOUT ROWID" in (create_sql or "").upper()
    names = [row[1] for row in columns]
    primary = [
        row[1] for row in sorted(columns, key=lambda item: item[5]) if row[5]
    ]
    return {
        "create_sql": create_sql,
        "columns": names,
        "primary_key": primary,
        "without_rowid": without_rowid,
        "rowid_name": None if without_rowid else rowid_name(names),
    }


def _pk_value_key(values: list) -> bytes:
    return canonical_key(values[0] if len(values) == 1 else values)


def _rowmap_name(table: str) -> str:
    return "rowmap_" + hashlib.sha256(table.encode("utf-8")).hexdigest()


def _compile_row_store(
    con: sqlite3.Connection,
    tables: list[tuple[str, str]],
    store: SegmentStore,
    spool: sqlite3.Connection,
) -> dict[str, dict]:
    table_manifest = {}
    for table, create_sql in tables:
        schema = _table_schema(con, table, create_sql or "")
        columns = schema["columns"]
        primary = schema["primary_key"]
        table_hash = hashlib.sha256(table.encode("utf-8")).hexdigest()[:16]
        rows_name = f"table:{table_hash}:rows"
        offsets_name = f"table:{table_hash}:offsets"
        rowids_name = f"table:{table_hash}:rowids"
        rows_writer = store.writer(rows_name, "uint8")
        offsets_writer = store.writer(offsets_name, "<i8")
        rowids_writer = store.writer(rowids_name, "<i8")
        count = 0
        byte_offset = 0
        if schema["without_rowid"]:
            if not primary:
                raise ValueError(f"WITHOUT ROWID table {table!r} has no primary key")
            selected_columns = ", ".join(quote_identifier(col) for col in columns)
            order = ", ".join(quote_identifier(col) for col in primary)
            cursor = con.execute(
                f"SELECT {selected_columns} FROM {quote_identifier(table)} ORDER BY {order}"
            )
            map_name = _rowmap_name(table)
            spool.execute(
                f'CREATE TABLE "{map_name}" (pk BLOB PRIMARY KEY, logical_id INTEGER NOT NULL) WITHOUT ROWID'
            )
            map_insert = f'INSERT INTO "{map_name}" (pk, logical_id) VALUES (?, ?)'
            for row in cursor:
                row_values = list(row)
                logical_id = count + 1
                pk_values = [row_values[columns.index(col)] for col in primary]
                spool.execute(map_insert, (_pk_value_key(pk_values), logical_id))
                encoded = canonical_row(row_values)
                offsets_writer.write_i64(byte_offset)
                rowids_writer.write_i64(logical_id)
                rows_writer.write(encoded)
                byte_offset += len(encoded)
                count += 1
            spool.commit()
        else:
            rowid_column = schema["rowid_name"]
            selected_columns = ", ".join(quote_identifier(col) for col in columns)
            cursor = con.execute(
                f"SELECT {quote_identifier(rowid_column)}, {selected_columns} "
                f"FROM {quote_identifier(table)} ORDER BY {quote_identifier(rowid_column)}"
            )
            for row in cursor:
                logical_id = int(row[0])
                encoded = canonical_row(row[1:])
                offsets_writer.write_i64(byte_offset)
                rowids_writer.write_i64(logical_id)
                rows_writer.write(encoded)
                byte_offset += len(encoded)
                count += 1
        rows_writer.finish([byte_offset])
        offsets_writer.finish([count])
        rowids_writer.finish([count])
        schema.update(
            {
                "row_count": count,
                "segments": {
                    "rows": rows_name,
                    "offsets": offsets_name,
                    "rowids": rowids_name,
                },
            }
        )
        table_manifest[table] = schema
        status(f"Stored rows for {table}: {count} rows")
    return table_manifest


def _index_segments(
    con: sqlite3.Connection,
    spec: QuerySpec,
    table_schema: dict,
    store: SegmentStore,
    spool: sqlite3.Connection,
) -> tuple[list[str], int, int, int]:
    assert spec.table is not None
    table = spec.table
    columns = table_schema["columns"]
    no_rowid = table_schema["without_rowid"]
    key_columns = spec.key_columns
    segment_id = hashlib.sha256(spec.id.encode("utf-8")).hexdigest()[:16]
    key_blob_name = f"query:{segment_id}:keys"
    key_offsets_name = f"query:{segment_id}:key_offsets"
    indptr_name = f"query:{segment_id}:indptr"
    indices_name = f"query:{segment_id}:indices"
    map_name = _rowmap_name(table)
    if no_rowid:
        primary = table_schema["primary_key"]
        selected = list(primary)
        selected.extend(column for column in key_columns if column not in selected)
        select_sql = ", ".join(quote_identifier(col) for col in selected)
        order_sql = ", ".join(
            [quote_identifier(col) for col in key_columns]
            + [quote_identifier(col) for col in primary]
        )
        cursor = con.execute(
            f"SELECT {select_sql} FROM {quote_identifier(table)} ORDER BY {order_sql}"
        )
        selected_positions = {column: i for i, column in enumerate(selected)}
        rowmap_cursor = spool.cursor()
        rowid_expression = None
    else:
        rowid_column = table_schema["rowid_name"]
        select_sql = ", ".join(
            [quote_identifier(rowid_column)]
            + [quote_identifier(column) for column in key_columns]
        )
        order_sql = ", ".join(
            [quote_identifier(column) for column in key_columns]
            + [quote_identifier(rowid_column)]
        )
        cursor = con.execute(
            f"SELECT {select_sql} FROM {quote_identifier(table)} ORDER BY {order_sql}"
        )
        rowid_expression = rowid_column
        rowmap_cursor = None

    spool_name = "idx_" + segment_id
    spool.execute(
        f'DROP TABLE IF EXISTS "{spool_name}"'
    )
    spool.execute(
        f'CREATE TABLE "{spool_name}" (key BLOB NOT NULL, rowid INTEGER NOT NULL, PRIMARY KEY(key, rowid)) WITHOUT ROWID'
    )
    insert_sql = f'INSERT INTO "{spool_name}" (key, rowid) VALUES (?, ?)'
    null_keys = 0
    for row in cursor:
        if no_rowid:
            value_by_column = {name: row[pos] for name, pos in selected_positions.items()}
            key_values = [value_by_column[column] for column in key_columns]
            pk_values = [value_by_column[column] for column in table_schema["primary_key"]]
            map_row = rowmap_cursor.execute(
                f'SELECT logical_id FROM "{map_name}" WHERE pk = ?',
                (_pk_value_key(pk_values),),
            ).fetchone()
            if map_row is None:
                raise RuntimeError(f"Missing row identity for {table}")
            rowid = map_row[0]
        else:
            rowid = int(row[0])
            key_values = list(row[1:])
        if any(value is None for value in key_values):
            null_keys += 1
            continue
        if spec.kind == "text":
            key_value = str(key_values[0]).lower()
        else:
            key_value = key_values[0] if len(key_values) == 1 else key_values
        spool.execute(insert_sql, (canonical_key(key_value), rowid))
    spool.commit()

    keys_writer = store.writer(key_blob_name, "uint8")
    key_offsets_writer = store.writer(key_offsets_name, "<i8")
    indptr_writer = store.writer(indptr_name, "<i8")
    indices_writer = store.writer(indices_name, "<i8")
    key_offsets_writer.write_i64(0)
    indptr_writer.write_i64(0)
    key_count = 0
    row_count = 0
    key_bytes_count = 0
    previous_key = None
    for key, rowid in spool.execute(
        f'SELECT key, rowid FROM "{spool_name}" ORDER BY key, rowid'
    ):
        if key != previous_key:
            if previous_key is not None:
                indptr_writer.write_i64(row_count)
            keys_writer.write(key)
            key_bytes_count += len(key)
            key_count += 1
            key_offsets_writer.write_i64(key_bytes_count)
            previous_key = key
        indices_writer.write_i64(rowid)
        row_count += 1
    if key_count:
        indptr_writer.write_i64(row_count)
    keys_writer.finish([key_bytes_count])
    key_offsets_writer.finish([key_count + 1])
    indptr_writer.finish([key_count + 1])
    indices_writer.finish([row_count])
    spool.execute(f'DROP TABLE "{spool_name}"')
    spool.commit()
    return (
        [key_blob_name, key_offsets_name, indptr_name, indices_name],
        key_count,
        row_count,
        null_keys,
    )


def _result_segment(
    con: sqlite3.Connection, spec: QuerySpec, store: SegmentStore
) -> tuple[list[str], int]:
    segment_id = hashlib.sha256(spec.id.encode("utf-8")).hexdigest()[:16]
    result_name = f"query:{segment_id}:results"
    writer = store.writer(result_name, "uint8")
    writer.write(b"[")
    count = 0
    for row in con.execute(spec.sql):
        if count:
            writer.write(b",")
        writer.write(canonical_row(row))
        count += 1
    writer.write(b"]")
    writer.finish([writer.nbytes])
    return [result_name], count


def _write_operator_segments(
    con: sqlite3.Connection,
    workbook: Path,
    store: SegmentStore,
) -> dict[str, int]:
    E, _ = load_workbook(workbook)
    store.add_array("channel:E", E, "<f8")
    counts: dict[str, int] = {}
    for name in QUERIES:
        try:
            T, count = compile_channel(con, name, 0)
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower() or "no such column" in str(exc).lower():
                status(f"Skipped channel {name}: {exc}")
                continue
            raise
        B = direct_operator(E, T)
        store.add_array(f"channel:T:{name}", T, "<f8")
        store.add_array(f"channel:B:{name}", B, "<f8")
        counts[name] = count
        status(f"Compiled channel {name}: {count} rows")
    return counts


def compile_model(db: Path, workbook: Path, output: Path, only_channels: bool = False) -> dict:
    started = time.perf_counter()
    db = db.resolve()
    workbook = workbook.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status("START read-only SQL-IR compilation")
    db_hash = sha256_file(db)
    workbook_hash = sha256_file(workbook)
    uri = db.as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only=ON")
    status("Opened SQLite database read-only")

    with tempfile.TemporaryDirectory(prefix="sqlir-", dir=output) as temp_dir:
        temporary = Path(temp_dir)
        store = SegmentStore(temporary)
        tables = _database_tables(con)
        schemas = {table: sql for table, sql in tables}
        table_manifest = {}
        query_manifest = []
        spool = sqlite3.connect(temporary / "index-spool.sqlite")
        spool.execute("PRAGMA journal_mode=OFF")
        spool.execute("PRAGMA synchronous=OFF")
        spool.execute("PRAGMA temp_store=FILE")
        spool.execute("PRAGMA cache_size=-65536")

        if not only_channels:
            specs = build_catalog(con)
            table_manifest = _compile_row_store(con, tables, store, spool)
            for spec in specs:
                if spec.kind in {"pk", "eq", "fk", "text"}:
                    segment_names, key_count, row_count, null_keys = _index_segments(
                        con,
                        spec,
                        table_manifest[spec.table],
                        store,
                        spool,
                    )
                    n_keys = key_count
                    n_rows = row_count
                else:
                    segment_names, n_rows = _result_segment(con, spec, store)
                    null_keys = 0
                    if spec.kind == "count":
                        n_keys = n_rows
                    else:
                        n_keys = 0
                query_manifest.append(
                    {
                        "id": spec.id,
                        "kind": spec.kind,
                        "sql": spec.sql,
                        "sql_sha256": hashlib.sha256(
                            spec.sql.encode("utf-8")
                        ).hexdigest(),
                        "table": spec.table,
                        "key_columns": spec.key_columns,
                        "n_keys": n_keys,
                        "n_rows": n_rows,
                        "null_keys": null_keys,
                        "segments": segment_names,
                    }
                )
                status(
                    f"Compiled query {spec.id}: keys={n_keys}, rows={n_rows}, "
                    f"null_keys={null_keys}"
                )

        channel_counts = _write_operator_segments(con, workbook, store)
        spool.close()
        con.close()

        blob_path = output / "blob.bin"
        segments = {}
        with blob_path.open("wb") as blob:
            for name, descriptor in store.descriptors.items():
                padding = (-blob.tell()) % 64
                if padding:
                    blob.write(b"\0" * padding)
                offset = blob.tell()
                digest = hashlib.sha256()
                with store.paths[name].open("rb") as segment:
                    while chunk := segment.read(CHUNK_SIZE):
                        blob.write(chunk)
                        digest.update(chunk)
                segments[name] = {
                    "name": name,
                    **descriptor,
                    "offset": offset,
                    "sha256": digest.hexdigest(),
                }

        manifest = {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db),
            "db_sha256": db_hash,
            "db_size": db.stat().st_size,
            "workbook_path": str(workbook),
            "workbook_sha256": workbook_hash,
            "sqlite_version": sqlite3.sqlite_version,
            "compile_duration_seconds": time.perf_counter() - started,
            "only_channels": only_channels,
            "compiled_rows": channel_counts,
            "tables": table_manifest,
            "schemas": schemas,
            "queries": query_manifest,
            "segments": segments,
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    status(
        f"Finished model: {blob_path.stat().st_size} bytes; "
        f"manifest={output / 'manifest.json'}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only-channels", action="store_true")
    args = parser.parse_args()
    compile_model(args.db, args.workbook, args.out, args.only_channels)


if __name__ == "__main__":
    main()
