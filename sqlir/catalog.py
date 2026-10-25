"""Schema-derived query catalog for SQLite databases."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from .conical_sql_ir_transformer import QUERIES


TEXT_NAMES = {
    "lemma",
    "normalized",
    "form",
    "surface_form",
    "text",
    "headword",
    "word",
    "left_lemma",
    "right_lemma",
    "target_lemma",
    "sentence",
}


@dataclass(frozen=True)
class QuerySpec:
    id: str
    kind: str
    sql: str
    key_columns: list[str]
    table: str | None


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def json_value(value: Any) -> Any:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return {"$b64": base64.b64encode(bytes(value)).decode("ascii")}
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_row(values: Any) -> bytes:
    return canonical_json([json_value(value) for value in values])


def canonical_key(value: Any) -> bytes:
    return canonical_json(json_value(value))


def rowid_name(columns: list[str]) -> str:
    for candidate in ("_rowid_", "rowid", "oid"):
        if candidate not in columns:
            return candidate
    raise ValueError("Cannot address SQLite rowid because all aliases are shadowed")


def _table_columns(con: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return list(con.execute(f"PRAGMA table_info({quote_identifier(table)})"))


def _table_order(columns: list[sqlite3.Row], without_rowid: bool) -> list[str]:
    if without_rowid:
        return [
            row[1]
            for row in sorted(columns, key=lambda item: item[5])
            if row[5]
        ]
    return [rowid_name([row[1] for row in columns])]


def _result_sql(
    table: str,
    columns: list[sqlite3.Row],
    without_rowid: bool,
    key_columns: list[str],
    text: bool = False,
) -> str:
    where = []
    for column in key_columns:
        quoted = quote_identifier(column)
        where.append(f"sqlir_lower({quoted}) = ?" if text else f"{quoted} = ?")
    order = _table_order(columns, without_rowid)
    return (
        f"SELECT * FROM {quote_identifier(table)} WHERE "
        + " AND ".join(where)
        + " ORDER BY "
        + ", ".join(quote_identifier(column) for column in order)
    )


def _referenced_tables(sql: str, table_names: set[str]) -> set[str]:
    references = re.findall(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
        sql,
        flags=re.IGNORECASE,
    )
    lower_names = {name.lower(): name for name in table_names}
    if not references or any(name.lower() not in lower_names for name in references):
        return set()
    return {lower_names[name.lower()] for name in references}


def build_catalog(con: sqlite3.Connection) -> list[QuerySpec]:
    """Build stable query specs using only SQLite schema metadata and fixed channels."""
    tables = list(
        con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    )
    names = {row[0] for row in tables}
    specs: list[QuerySpec] = []
    for table, create_sql in tables:
        columns = _table_columns(con, table)
        create_sql = create_sql or ""
        without_rowid = bool(re.search(r"\bWITHOUT\s+ROWID\b", create_sql, re.I))
        declared_pk = [
            row[1]
            for row in sorted(columns, key=lambda item: item[5])
            if row[5]
        ]
        primary = declared_pk or [rowid_name([row[1] for row in columns])]
        specs.append(
            QuerySpec(
                id=f"pk:{table}",
                kind="pk",
                sql=_result_sql(table, columns, without_rowid, primary),
                key_columns=primary,
                table=table,
            )
        )

        indexed_columns: set[str] = set()
        for index in con.execute(f"PRAGMA index_list({quote_identifier(table)})"):
            for index_column in con.execute(
                f"PRAGMA index_info({quote_identifier(index[1])})"
            ):
                if index_column[0] == 0 and index_column[2] is not None:
                    indexed_columns.add(index_column[2])

        foreign_keys: dict[int, list[sqlite3.Row]] = {}
        for foreign_key in con.execute(
            f"PRAGMA foreign_key_list({quote_identifier(table)})"
        ):
            foreign_keys.setdefault(foreign_key[0], []).append(foreign_key)
            indexed_columns.add(foreign_key[3])

        for column in sorted(indexed_columns):
            specs.append(
                QuerySpec(
                    id=f"eq:{table}.{column}",
                    kind="eq",
                    sql=_result_sql(table, columns, without_rowid, [column]),
                    key_columns=[column],
                    table=table,
                )
            )

        for foreign_id in sorted(foreign_keys):
            members = sorted(foreign_keys[foreign_id], key=lambda item: item[1])
            parent = members[0][2]
            parent_columns = [member[4] for member in members]
            if any(column is None for column in parent_columns):
                parent_info = _table_columns(con, parent)
                parent_columns = [
                    row[1]
                    for row in sorted(parent_info, key=lambda item: item[5])
                    if row[5]
                ]
            child_columns = [member[3] for member in members]
            description = ",".join(child_columns)
            target = ",".join(parent_columns)
            specs.append(
                QuerySpec(
                    id=f"fk:{table}.{description}->{parent}.{target}",
                    kind="fk",
                    sql=_result_sql(
                        table, columns, without_rowid, child_columns
                    ),
                    key_columns=child_columns,
                    table=table,
                )
            )

        for column in sorted(row[1] for row in columns):
            info = next(row for row in columns if row[1] == column)
            declared_type = (info[2] or "").upper()
            text_affinity = (
                "CHAR" in declared_type
                or "CLOB" in declared_type
                or "TEXT" in declared_type
            )
            if text_affinity and (
                column.lower() in TEXT_NAMES
                or column.lower().endswith("_lemma")
                or column.lower().endswith("_form")
            ):
                specs.append(
                    QuerySpec(
                        id=f"text:{table}.{column}",
                        kind="text",
                        sql=_result_sql(
                            table, columns, without_rowid, [column], text=True
                        ),
                        key_columns=[column],
                        table=table,
                    )
                )

        primary_columns = {row[1] for row in columns if row[5]}
        for column in sorted(row[1] for row in columns):
            info = next(row for row in columns if row[1] == column)
            declared_type = (info[2] or "").upper()
            if (
                column in primary_columns
                or "REAL" in declared_type
                or "FLOA" in declared_type
                or "DOUB" in declared_type
            ):
                continue
            quoted = quote_identifier(column)
            distinct = con.execute(
                f"SELECT COUNT(DISTINCT {quoted}) FROM {quote_identifier(table)}"
            ).fetchone()[0]
            if distinct <= 5000:
                specs.append(
                    QuerySpec(
                        id=f"count:{table}.{column}",
                        kind="count",
                        sql=(
                            f"SELECT {quoted}, COUNT(*) FROM {quote_identifier(table)} "
                            f"GROUP BY {quoted} ORDER BY {quoted}"
                        ),
                        key_columns=[column],
                        table=table,
                    )
                )

    for name, sql in QUERIES.items():
        referenced = _referenced_tables(sql, names)
        if referenced and referenced.issubset(names):
            specs.append(
                QuerySpec(
                    id=f"fixed:conical.{name}",
                    kind="fixed",
                    sql=sql,
                    key_columns=[],
                    table=None,
                )
            )
    return specs
