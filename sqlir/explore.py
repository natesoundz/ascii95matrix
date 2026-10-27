"""Continuous, resumable, read-only discovery over the Webster lexicon SQLite.

Probes are derived from the live schema (every table, column, view and foreign
key) plus a fixed family for the ``sx_`` syllable layer.  Each probe's result is
hashed with SHA-256 and written to a separate ledger SQLite file; analysers turn
results into findings and new child probes, so exploration keeps expanding
until the frontier is exhausted (and, with ``--watch``, restarts when the
source database changes).  The source database is never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .catalog import canonical_json, canonical_row, json_value, quote_identifier
from .conical_sql_ir_transformer import UNRESOLVED


STORED_TEXT_LIMIT = 300
HASH_CHUNK = 8 * 1024 * 1024
STRESS_MARKS = {'"': "primary", "`": "secondary", "*": "break"}


@dataclass(frozen=True)
class Probe:
    id: str
    family: str
    title: str
    sql: str = ""
    params: tuple = ()
    py: str = ""
    priority: int = 50
    depth: int = 0
    parent: str | None = None
    store_limit: int = 100
    expect_zero: bool = False
    sample_sql: str = ""


@dataclass
class Finding:
    kind: str
    subject: str
    summary: str
    evidence: Any = None


@dataclass
class Context:
    max_depth: int
    fanout: int
    children: list[Probe] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def child(self, parent: Probe, probe: Probe) -> None:
        if parent.depth + 1 > self.max_depth:
            return
        self.children.append(
            Probe(
                **{
                    **probe.__dict__,
                    "depth": parent.depth + 1,
                    "parent": parent.id,
                    "priority": max(probe.priority, parent.priority + 1),
                }
            )
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def q(name: str) -> str:
    return quote_identifier(name)


def stored_value(value: Any) -> Any:
    value = json_value(value)
    if isinstance(value, str) and len(value) > STORED_TEXT_LIMIT:
        return value[:STORED_TEXT_LIMIT] + f"...[+{len(value) - STORED_TEXT_LIMIT} chars]"
    return value


def open_source(path: Path) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only=ON")
    return con


def file_sha256(path: Path, log: Callable[[str], None]) -> str:
    digest = hashlib.sha256()
    size = max(path.stat().st_size, 1)
    done = 0
    started = time.monotonic()
    last = started
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last > 5:
                rate = done / max(now - started, 1e-9)
                log(f"hashing {path.name}: {done / size:.0%}, eta {(size - done) / rate:.0f}s")
                last = now
    return digest.hexdigest()


def schema_sha256(con: sqlite3.Connection) -> str:
    rows = con.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    return hashlib.sha256(canonical_json([list(row) for row in rows])).hexdigest()


def stat_signature(path: Path) -> dict:
    wal = Path(str(path) + "-wal")
    st = path.stat()
    signature = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "wal_size": 0, "wal_mtime_ns": 0}
    if wal.exists():
        wst = wal.stat()
        signature.update(wal_size=wst.st_size, wal_mtime_ns=wst.st_mtime_ns)
    return signature


# ---------------------------------------------------------------- ledger


LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_state(
  state_id INTEGER PRIMARY KEY, db_path TEXT NOT NULL, size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL, wal_size INTEGER NOT NULL, wal_mtime_ns INTEGER NOT NULL,
  sha256 TEXT NOT NULL, schema_sha256 TEXT NOT NULL, first_seen TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS probe(
  state_id INTEGER NOT NULL, probe_id TEXT NOT NULL, family TEXT NOT NULL,
  title TEXT NOT NULL, sql TEXT NOT NULL, params_json TEXT NOT NULL, py TEXT NOT NULL,
  priority INTEGER NOT NULL, depth INTEGER NOT NULL, parent_id TEXT,
  store_limit INTEGER NOT NULL, expect_zero INTEGER NOT NULL, sample_sql TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', PRIMARY KEY(state_id, probe_id));
CREATE INDEX IF NOT EXISTS probe_frontier ON probe(state_id, status, priority, depth, probe_id);
CREATE TABLE IF NOT EXISTS result(
  state_id INTEGER NOT NULL, probe_id TEXT NOT NULL, started_at TEXT NOT NULL,
  elapsed_s REAL NOT NULL, row_count INTEGER, columns_json TEXT, rows_json TEXT,
  result_sha256 TEXT, error TEXT, PRIMARY KEY(state_id, probe_id));
CREATE TABLE IF NOT EXISTS finding(
  state_id INTEGER NOT NULL, probe_id TEXT NOT NULL, kind TEXT NOT NULL,
  subject TEXT NOT NULL, summary TEXT NOT NULL, evidence_json TEXT,
  PRIMARY KEY(state_id, probe_id, kind, subject));
"""


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(LEDGER_SCHEMA)

    def state_for(self, db: Path, src: sqlite3.Connection, log: Callable[[str], None]) -> tuple[int, bool]:
        db = Path(db).resolve()
        sig = stat_signature(db)
        schema = schema_sha256(src)
        row = self.con.execute(
            "SELECT state_id FROM source_state WHERE db_path=? AND size=? AND mtime_ns=? "
            "AND wal_size=? AND wal_mtime_ns=? AND schema_sha256=? ORDER BY state_id DESC LIMIT 1",
            (str(db), sig["size"], sig["mtime_ns"], sig["wal_size"], sig["wal_mtime_ns"], schema),
        ).fetchone()
        if row:
            return row[0], False
        sha = file_sha256(db, log)
        cur = self.con.execute(
            "INSERT INTO source_state(db_path,size,mtime_ns,wal_size,wal_mtime_ns,sha256,schema_sha256,first_seen) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (str(db), sig["size"], sig["mtime_ns"], sig["wal_size"], sig["wal_mtime_ns"], sha, schema, utc_now()),
        )
        self.con.commit()
        return cur.lastrowid, True

    def add_probes(self, state: int, probes: Iterable[Probe]) -> int:
        before = self.con.total_changes
        self.con.executemany(
            "INSERT OR IGNORE INTO probe(state_id,probe_id,family,title,sql,params_json,py,priority,depth,"
            "parent_id,store_limit,expect_zero,sample_sql) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    state, p.id, p.family, p.title, p.sql, json.dumps(list(p.params)), p.py,
                    p.priority, p.depth, p.parent, p.store_limit, int(p.expect_zero), p.sample_sql,
                )
                for p in probes
            ],
        )
        self.con.commit()
        return self.con.total_changes - before

    def next_pending(self, state: int) -> Probe | None:
        row = self.con.execute(
            "SELECT probe_id,family,title,sql,params_json,py,priority,depth,parent_id,store_limit,"
            "expect_zero,sample_sql FROM probe WHERE state_id=? AND status='pending' "
            "ORDER BY priority, depth, probe_id LIMIT 1",
            (state,),
        ).fetchone()
        if row is None:
            return None
        return Probe(
            id=row[0], family=row[1], title=row[2], sql=row[3], params=tuple(json.loads(row[4])),
            py=row[5], priority=row[6], depth=row[7], parent=row[8], store_limit=row[9],
            expect_zero=bool(row[10]), sample_sql=row[11],
        )

    def record(self, state: int, probe: Probe, outcome: dict, findings: list[Finding]) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO result(state_id,probe_id,started_at,elapsed_s,row_count,columns_json,"
            "rows_json,result_sha256,error) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                state, probe.id, outcome["started_at"], outcome["elapsed_s"], outcome.get("row_count"),
                json.dumps(outcome.get("columns")), json.dumps(outcome.get("rows"), ensure_ascii=False),
                outcome.get("sha256"), outcome.get("error"),
            ),
        )
        self.con.executemany(
            "INSERT OR REPLACE INTO finding(state_id,probe_id,kind,subject,summary,evidence_json) VALUES(?,?,?,?,?,?)",
            [
                (state, probe.id, f.kind, f.subject, f.summary, json.dumps(f.evidence, ensure_ascii=False))
                for f in findings
            ],
        )
        self.con.execute(
            "UPDATE probe SET status=? WHERE state_id=? AND probe_id=?",
            (outcome["status"], state, probe.id),
        )
        self.con.commit()

    def counts(self, state: int) -> dict[str, int]:
        return dict(
            self.con.execute(
                "SELECT status, COUNT(*) FROM probe WHERE state_id=? GROUP BY status", (state,)
            ).fetchall()
        )

    def close(self) -> None:
        self.con.close()


# ---------------------------------------------------------------- execution


def run_probe(con: sqlite3.Connection, probe: Probe, timeout_s: float) -> dict:
    started = time.monotonic()
    deadline = started + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 20000)
    outcome: dict[str, Any] = {"started_at": utc_now()}
    try:
        if probe.py:
            columns, rows = PY_PROBES[probe.py](con, probe.params)
        else:
            cursor = con.execute(probe.sql, probe.params)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor
        digest = hashlib.sha256(canonical_json(columns))
        stored: list[list[Any]] = []
        count = 0
        for row in rows:
            encoded = canonical_row(row)
            digest.update(encoded)
            digest.update(b"\n")
            if count < probe.store_limit:
                stored.append([stored_value(v) for v in row])
            count += 1
            if time.monotonic() > deadline:
                raise sqlite3.OperationalError("interrupted")
        outcome.update(
            status="done", columns=columns, rows=stored, row_count=count, sha256=digest.hexdigest()
        )
    except sqlite3.OperationalError as exc:
        status = "timeout" if "interrupt" in str(exc).lower() else "error"
        outcome.update(status=status, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - record and keep exploring
        outcome.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        con.set_progress_handler(None, 0)
    outcome["elapsed_s"] = round(time.monotonic() - started, 3)
    return outcome


def analyse(probe: Probe, outcome: dict, ctx: Context) -> None:
    if outcome["status"] != "done":
        ctx.findings.append(Finding(outcome["status"], probe.id, outcome.get("error", ""), None))
        return
    rows = outcome["rows"]
    if probe.expect_zero:
        value = rows[0][0] if rows and rows[0] else None
        if value == 0:
            ctx.findings.append(Finding("invariant_pass", probe.id, f"{probe.title}: 0 violations", 0))
        else:
            ctx.findings.append(
                Finding("invariant_fail", probe.id, f"{probe.title}: {value} violations", value)
            )
            if probe.sample_sql:
                ctx.child(
                    probe,
                    Probe(
                        id=f"sample:{probe.id}", family="sample",
                        title=f"Examples violating: {probe.title}", sql=probe.sample_sql,
                        params=probe.params, priority=probe.priority + 1, store_limit=50,
                    ),
                )
    analyser = ANALYSERS.get(probe.family)
    if analyser:
        analyser(probe, outcome, ctx)


# ---------------------------------------------------------------- schema-derived probes


def schema_probes(con: sqlite3.Connection) -> list[Probe]:
    probes: list[Probe] = []
    objects = con.execute(
        "SELECT type, name FROM sqlite_master WHERE type IN ('table','view') "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    tables = {name for kind, name in objects if kind == "table"}
    for kind, name in objects:
        if kind == "view":
            probes.append(
                Probe(
                    id=f"view.sample:{name}", family="view.sample", title=f"First rows of view {name}",
                    sql=f"SELECT * FROM {q(name)} LIMIT 5", priority=45,
                )
            )
            continue
        probes.append(
            Probe(
                id=f"table.count:{name}", family="table.count", title=f"Row count of {name}",
                sql=f"SELECT COUNT(*) FROM {q(name)}", params=(), priority=20,
            )
        )
        for _cid, column, *_ in con.execute(f"PRAGMA table_info({q(name)})"):
            c = q(column)
            probes.append(
                Probe(
                    id=f"column.profile:{name}.{column}", family="column.profile",
                    title=f"Profile of {name}.{column}",
                    sql=(
                        f"SELECT COUNT(*) AS total, COUNT({c}) AS non_null, COUNT(DISTINCT {c}) AS distinct_values, "
                        f"MIN({c}) AS min_value, MAX({c}) AS max_value, MIN(length({c})) AS min_length, "
                        f"MAX(length({c})) AS max_length FROM {q(name)}"
                    ),
                    priority=40,
                )
            )
        foreign: dict[int, list] = {}
        for fk in con.execute(f"PRAGMA foreign_key_list({q(name)})"):
            foreign.setdefault(fk[0], []).append(fk)
        for _fk_id, members in sorted(foreign.items()):
            members.sort(key=lambda m: m[1])
            parent = members[0][2]
            child_cols = [m[3] for m in members]
            parent_cols = [m[4] for m in members]
            subject = f"{name}.{','.join(child_cols)}->{parent}"
            if parent not in tables:
                probes.append(
                    Probe(
                        id=f"fk.parent:{subject}", family="fk.parent",
                        title=f"FK parent table {parent} exists", sql="SELECT 1", priority=30,
                        expect_zero=True,
                    )
                )
                continue
            if any(col is None for col in parent_cols):
                pk = [
                    r[1]
                    for r in sorted(con.execute(f"PRAGMA table_info({q(parent)})"), key=lambda r: r[5])
                    if r[5]
                ]
                parent_cols = pk if len(pk) == len(child_cols) else ["rowid"]
            not_null = " AND ".join(f"c.{q(col)} IS NOT NULL" for col in child_cols)
            match = " AND ".join(
                f"p.{q(pc)} = c.{q(cc)}" for pc, cc in zip(parent_cols, child_cols)
            )
            where = f"WHERE {not_null} AND NOT EXISTS (SELECT 1 FROM {q(parent)} p WHERE {match})"
            probes.append(
                Probe(
                    id=f"fk.orphans:{subject}", family="fk.orphans",
                    title=f"Orphan rows {subject}",
                    sql=f"SELECT COUNT(*) FROM {q(name)} c {where}", priority=30, expect_zero=True,
                    sample_sql=f"SELECT c.* FROM {q(name)} c {where} LIMIT 20",
                )
            )
    return probes


def analyse_count(probe: Probe, outcome: dict, ctx: Context) -> None:
    table = probe.id.split(":", 1)[1]
    if outcome["rows"] and outcome["rows"][0][0] == 0:
        ctx.findings.append(Finding("empty_table", table, f"{table} has 0 rows", 0))


def analyse_profile(probe: Probe, outcome: dict, ctx: Context) -> None:
    subject = probe.id.split(":", 1)[1]
    table, column = subject.split(".", 1)
    total, non_null, distinct, *_ = outcome["rows"][0]
    evidence = dict(zip(outcome["columns"], outcome["rows"][0]))
    if total == 0:
        return
    if non_null == 0:
        ctx.findings.append(Finding("all_null", subject, f"{subject} is NULL in all {total} rows", evidence))
        return
    if distinct == 1:
        ctx.findings.append(
            Finding("constant", subject, f"{subject} has a single value across {non_null} rows", evidence)
        )
    if distinct == non_null == total:
        ctx.findings.append(Finding("unique", subject, f"{subject} is unique and non-null ({total} rows)", evidence))
    if non_null < total:
        ctx.findings.append(
            Finding("partial_null", subject, f"{subject} is NULL in {total - non_null} of {total} rows", evidence)
        )
    if 1 < distinct < non_null:
        c = q(column)
        ctx.child(
            probe,
            Probe(
                id=f"column.top:{subject}", family="column.top", title=f"Most frequent values of {subject}",
                sql=(
                    f"SELECT {c} AS value, COUNT(*) AS n FROM {q(table)} GROUP BY {c} "
                    f"ORDER BY n DESC, {c} LIMIT 25"
                ),
                priority=60, store_limit=25,
            ),
        )


# ---------------------------------------------------------------- syllable layer

TOKENS = """
SELECT ws.variant_id, ws.ordinal, ws.start_char, ws.end_char, s.syllable_id, s.text,
       COUNT(*) OVER (PARTITION BY ws.variant_id) AS n,
       CASE WHEN COUNT(*) OVER (PARTITION BY ws.variant_id) = 1 THEN 'sole'
            WHEN ws.ordinal = 0 THEN 'initial'
            WHEN ws.ordinal = COUNT(*) OVER (PARTITION BY ws.variant_id) - 1 THEN 'final'
            ELSE 'medial' END AS position
FROM sx_word_syllable ws JOIN sx_syllable s USING(syllable_id)
"""

LETTER_SLOTS = """
SELECT sc.codepoint,
       CASE WHEN length(s.text) = 1 THEN 'only'
            WHEN sc.ordinal = 0 THEN 'first'
            WHEN sc.ordinal = length(s.text) - 1 THEN 'last'
            ELSE 'interior' END AS slot,
       COUNT(*) AS tokens
FROM sx_word_syllable ws JOIN sx_syllable s USING(syllable_id)
JOIN sx_syllable_character sc USING(syllable_id)
GROUP BY 1, 2
"""

SPAN_GAPS = """
FROM (SELECT variant_id, ordinal, start_char, end_char,
             LAG(end_char) OVER w AS prev_end, LAG(ordinal) OVER w AS prev_ordinal
      FROM sx_word_syllable WINDOW w AS (PARTITION BY variant_id ORDER BY ordinal))
WHERE (prev_ordinal IS NULL AND (ordinal <> 0 OR start_char <> 0))
   OR (prev_ordinal IS NOT NULL AND (ordinal <> prev_ordinal + 1 OR start_char <> prev_end))
"""

SPAN_TEXT = """
FROM sx_word_syllable ws JOIN sx_syllable s USING(syllable_id) JOIN sx_word_variant v USING(variant_id)
WHERE ws.end_char - ws.start_char <> length(s.text)
   OR substr(v.spelling, ws.start_char + 1, ws.end_char - ws.start_char) <> s.text
"""

SPAN_COVER = """
FROM (SELECT v.variant_id, v.spelling, length(v.spelling) AS spelling_length, MAX(ws.end_char) AS covered
      FROM sx_word_variant v JOIN sx_word_syllable ws USING(variant_id) GROUP BY v.variant_id)
WHERE covered <> spelling_length
"""

RECORD_BODY = "substr(source_record, instr(source_record, '=') + 1)"
RECORD_STRIPPED = f"replace(replace(replace({RECORD_BODY}, '*', ''), '\"', ''), '`', '')"

CHAR_ROWS = """
FROM sx_syllable_character sc JOIN sx_syllable s USING(syllable_id)
WHERE unicode(substr(s.text, sc.ordinal + 1, 1)) IS NOT sc.codepoint
"""

CHAR_COUNT = """
FROM sx_syllable s WHERE s.kind = 'orthographic'
AND length(s.text) <> (SELECT COUNT(*) FROM sx_syllable_character sc WHERE sc.syllable_id = s.syllable_id)
"""

NUCLEUS = """
FROM sx_syllable s WHERE s.kind = 'phonetic' AND s.analysis_status = 'source_reported'
AND (SELECT COUNT(*) FROM sx_phone p WHERE p.syllable_id = s.syllable_id AND p.role = 'nucleus') <> 1
"""

ROLE_ORDER = """
FROM sx_phone a JOIN sx_phone b ON a.syllable_id = b.syllable_id AND b.ordinal > a.ordinal
WHERE (a.role = 'coda' AND b.role IN ('onset', 'nucleus')) OR (a.role = 'nucleus' AND b.role = 'onset')
"""

WORD_SYLLABLE_KIND = """
FROM sx_word_syllable ws JOIN sx_syllable s USING(syllable_id) WHERE s.kind <> 'orthographic'
"""

VARIANT_WITHOUT_SYLLABLES = """
FROM sx_word_variant v WHERE NOT EXISTS (SELECT 1 FROM sx_word_syllable ws WHERE ws.variant_id = v.variant_id)
"""

PRIMARY_PRON = """
SELECT lexeme_id, MIN(pronunciation_id) AS pronunciation_id FROM pronunciation
WHERE syllable_count IS NOT NULL GROUP BY lexeme_id
"""


def invariant(pid: str, title: str, body: str, sample_select: str, priority: int = 5) -> Probe:
    return Probe(
        id=f"syll.inv:{pid}", family="syll.invariant", title=title,
        sql=f"SELECT COUNT(*) {body}", priority=priority, expect_zero=True,
        sample_sql=f"SELECT {sample_select} {body} LIMIT 20",
    )


def syllable_probes(tables: set[str]) -> list[Probe]:
    required = {"sx_syllable", "sx_word_syllable", "sx_word_variant", "sx_syllable_character", "lexeme"}
    if not required.issubset(tables):
        return []
    probes = [
        invariant("span_gaps", "Syllable spans are contiguous from character 0 with consecutive ordinals",
                  SPAN_GAPS, "*"),
        invariant("span_text", "Each syllable span equals its syllable text in the spelling",
                  SPAN_TEXT, "v.variant_id, v.spelling, ws.ordinal, ws.start_char, ws.end_char, s.text"),
        invariant("span_cover", "Syllable spans cover the whole spelling", SPAN_COVER, "*"),
        invariant("variant_without_syllables", "Every word variant has syllables",
                  VARIANT_WITHOUT_SYLLABLES, "v.*"),
        invariant("word_syllable_kind", "Word syllables reference orthographic syllable types only",
                  WORD_SYLLABLE_KIND, "ws.*, s.kind, s.text"),
        invariant("record_reconstructs", "Source record without * \" ` marks equals the spelling (case-folded)",
                  f"FROM sx_word_variant WHERE upper({RECORD_STRIPPED}) <> upper(spelling)",
                  "variant_id, spelling, source_record"),
        invariant("record_headword", "Source record headword equals upper(spelling)",
                  "FROM sx_word_variant WHERE substr(source_record, 1, instr(source_record, '=') - 1) <> upper(spelling)",
                  "variant_id, spelling, source_record"),
        invariant("char_rows", "Syllable character rows match syllable text code points",
                  CHAR_ROWS, "s.syllable_id, s.kind, s.text, sc.ordinal, sc.codepoint"),
        invariant("char_count", "Orthographic syllables have one character row per letter",
                  CHAR_COUNT, "s.*"),
        Probe(
            id="syll.inventory", family="syll.observation", title="Syllable type inventory",
            sql=(
                "SELECT kind, notation, source_key, analysis_status, COUNT(*) AS types FROM sx_syllable "
                "GROUP BY 1,2,3,4 ORDER BY 1,2,3,4"
            ),
            priority=10,
        ),
        Probe(
            id="syll.coverage", family="syll.observation", title="Syllabification coverage across sources",
            sql=(
                "SELECT (SELECT COUNT(*) FROM lexeme) AS lexemes, "
                "(SELECT COUNT(DISTINCT lexeme_id) FROM sx_word_variant) AS syllabified_lexemes, "
                "(SELECT COUNT(*) FROM sx_word_variant) AS syllabified_variants, "
                + (
                    "(SELECT COUNT(DISTINCT lexeme_id) FROM pronunciation) AS cmu_lexemes, "
                    "(SELECT COUNT(*) FROM (SELECT lexeme_id FROM sx_word_variant INTERSECT "
                    "SELECT lexeme_id FROM pronunciation)) AS syllabified_and_cmu, "
                    if "pronunciation" in tables else ""
                )
                + (
                    "(SELECT COUNT(DISTINCT lexeme_id) FROM webster_lexeme) AS webster_lexemes, "
                    "(SELECT COUNT(*) FROM (SELECT lexeme_id FROM sx_word_variant INTERSECT "
                    "SELECT lexeme_id FROM webster_lexeme)) AS syllabified_and_webster, "
                    if "webster_lexeme" in tables else ""
                )
                + "(SELECT COUNT(*) FROM sx_word_syllable) AS syllable_tokens"
            ),
            priority=10,
        ),
        Probe(
            id="syll.multi_variant", family="syll.observation",
            title="Lexemes with more than one syllabified variant",
            sql=(
                "SELECT v.lexeme_id, l.lemma, COUNT(*) AS variants, group_concat(v.source_record, ' | ') AS records "
                "FROM sx_word_variant v JOIN lexeme l USING(lexeme_id) GROUP BY v.lexeme_id HAVING COUNT(*) > 1 "
                "ORDER BY variants DESC, l.lemma"
            ),
            priority=12, store_limit=200,
        ),
        Probe(
            id="syll.per_word", family="syll.observation", title="Syllables per word variant",
            sql=(
                "SELECT n AS syllables, COUNT(*) AS variants FROM "
                "(SELECT COUNT(*) AS n FROM sx_word_syllable GROUP BY variant_id) GROUP BY n ORDER BY n"
            ),
            priority=10,
        ),
        Probe(
            id="syll.length_by_position", family="syll.observation",
            title="Syllable letter length by position in word",
            sql=(
                f"SELECT position, end_char - start_char AS letters, COUNT(*) AS tokens FROM ({TOKENS}) "
                "GROUP BY 1, 2 ORDER BY 1, 2"
            ),
            priority=11, store_limit=500,
        ),
        Probe(
            id="syll.top_tokens", family="syll.top_tokens", title="Most frequent orthographic syllables (case-folded)",
            sql=(
                "SELECT lower(s.text) AS syllable, COUNT(*) AS tokens, COUNT(DISTINCT v.lexeme_id) AS lexemes, "
                "COUNT(DISTINCT s.syllable_id) AS case_types FROM sx_word_syllable ws "
                "JOIN sx_syllable s USING(syllable_id) JOIN sx_word_variant v USING(variant_id) "
                "GROUP BY 1 ORDER BY tokens DESC, syllable LIMIT 200"
            ),
            priority=11, store_limit=200,
        ),
        Probe(
            id="syll.case_types", family="syll.observation",
            title="Orthographic syllable types that differ only by case",
            sql=(
                "SELECT types_per_fold, COUNT(*) AS folds FROM (SELECT lower(text) AS fold, COUNT(*) AS types_per_fold "
                "FROM sx_syllable WHERE kind='orthographic' GROUP BY 1) GROUP BY 1 ORDER BY 1"
            ),
            priority=12,
        ),
        Probe(
            id="syll.boundary_bigrams", family="syll.boundary_bigrams",
            title="Letter pairs split by a syllable boundary (case-folded)",
            sql=(
                "SELECT lower(substr(v.spelling, a.end_char, 1) || substr(v.spelling, a.end_char + 1, 1)) AS bigram, "
                "COUNT(*) AS boundaries FROM sx_word_syllable a JOIN sx_word_syllable b "
                "ON b.variant_id = a.variant_id AND b.ordinal = a.ordinal + 1 "
                "JOIN sx_word_variant v ON v.variant_id = a.variant_id "
                "GROUP BY 1 ORDER BY boundaries DESC, bigram LIMIT 100"
            ),
            priority=13, store_limit=100,
        ),
        Probe(
            id="syll.letter_slots", family="syll.observation",
            title="Letter tokens by slot inside orthographic syllables",
            sql=(
                f"SELECT x.codepoint, c.character, x.slot, x.tokens FROM ({LETTER_SLOTS}) x "
                "LEFT JOIN sx_character c USING(codepoint) ORDER BY x.codepoint, x.slot"
            ),
            priority=14, store_limit=1000,
        ),
    ]
    if "pronunciation" in tables:
        probes += [
            Probe(
                id="syll.count_vs_cmu", family="syll.count_vs_cmu",
                title="Webster syllable count vs CMU syllable count (first CMU pronunciation per lexeme)",
                sql=(
                    "WITH w AS (SELECT v.variant_id, v.lexeme_id, COUNT(*) AS n FROM sx_word_variant v "
                    f"JOIN sx_word_syllable ws USING(variant_id) GROUP BY v.variant_id), pp AS ({PRIMARY_PRON}) "
                    "SELECT w.n AS webster_syllables, p.syllable_count AS cmu_syllables, COUNT(*) AS variants "
                    "FROM w JOIN pp USING(lexeme_id) JOIN pronunciation p ON p.pronunciation_id = pp.pronunciation_id "
                    "GROUP BY 1, 2 ORDER BY variants DESC, 1, 2"
                ),
                priority=15, store_limit=500,
            ),
            Probe(
                id="syll.stress_vs_cmu", family="syll.stress_vs_cmu", py="stress_vs_cmu",
                title="Webster primary-stress syllable vs CMU primary stress",
                priority=16, store_limit=2000,
            ),
        ]
    if "sx_character_attribute" in tables and "sx_attribute" in tables:
        probes += [
            Probe(
                id="wb.section_codes_by_slot", family="wb.section_codes",
                title="Canonical 437 workbook cell codes weighted by letter tokens in syllables, per section and slot",
                sql=(
                    f"WITH ls AS ({LETTER_SLOTS}) SELECT a.section, ls.slot, ca.value_json, "
                    "SUM(ls.tokens) AS letter_tokens, COUNT(DISTINCT ca.attribute_id) AS attributes "
                    "FROM ls JOIN sx_character_attribute ca USING(codepoint) "
                    "JOIN sx_attribute a USING(attribute_id) GROUP BY 1, 2, 3 ORDER BY 1, 2, letter_tokens DESC, 3"
                ),
                priority=17, store_limit=20000,
            ),
            Probe(
                id="wb.attribute_codes", family="wb.attribute_codes",
                title="Per-attribute workbook codes weighted by letter tokens in syllables",
                sql=(
                    f"WITH ls AS (SELECT codepoint, SUM(tokens) AS tokens FROM ({LETTER_SLOTS}) GROUP BY 1) "
                    "SELECT a.section, ca.attribute_id, a.is_physical, ca.value_json, SUM(ls.tokens) AS letter_tokens, "
                    "COUNT(*) AS letters FROM ls JOIN sx_character_attribute ca USING(codepoint) "
                    "JOIN sx_attribute a USING(attribute_id) GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, letter_tokens DESC, 4"
                ),
                priority=18, store_limit=50000,
            ),
            Probe(
                id="wb.letters_without_attributes", family="syll.invariant",
                title="Letters occurring in syllables have workbook attribute rows",
                sql=(
                    "SELECT COUNT(DISTINCT sc.codepoint) FROM sx_syllable_character sc WHERE NOT EXISTS "
                    "(SELECT 1 FROM sx_character_attribute ca WHERE ca.codepoint = sc.codepoint)"
                ),
                sample_sql=(
                    "SELECT DISTINCT sc.codepoint FROM sx_syllable_character sc WHERE NOT EXISTS "
                    "(SELECT 1 FROM sx_character_attribute ca WHERE ca.codepoint = sc.codepoint) LIMIT 20"
                ),
                priority=6, expect_zero=True,
            ),
        ]
    if "sx_phone" in tables:
        probes += [
            invariant("phonetic_nucleus", "Source-reported phonetic syllables have exactly one nucleus",
                      NUCLEUS, "s.*", priority=7),
            invariant("phone_role_order", "Phone roles are ordered onset < nucleus < coda",
                      ROLE_ORDER, "a.syllable_id, a.ordinal, a.role, b.ordinal, b.role", priority=7),
            Probe(
                id="phon.structure", family="syll.observation", py="phonetic_structure",
                title="Phonetic syllable shapes, onsets, nuclei and codas (BEEP)", priority=19, store_limit=3000,
            ),
        ]
        if "pronunciation" in tables:
            probes.append(
                Probe(
                    id="phon.cmu_inline_comments", family="syll.observation",
                    title="CMU pronunciations carrying an inline '#' comment in phones (excluded from phone counts)",
                    sql=(
                        "SELECT p.pronunciation_id, l.normalized, p.phones FROM pronunciation p "
                        "JOIN lexeme l USING(lexeme_id) WHERE instr(p.phones, '#') > 0 ORDER BY 1"
                    ),
                    priority=19, store_limit=500,
                )
            )
            probes.append(
                Probe(
                    id="phon.inventory_vs_cmu", family="syll.observation", py="phone_inventory",
                    title="Phone symbol inventory: BEEP syllables vs CMU pronunciations vs workbook phoneme sheet",
                    priority=19, store_limit=500,
                )
            )
    if "sx_candidate_example" in tables:
        probes.append(
            Probe(
                id="phon.candidate_links", family="syll.observation",
                title="BEEP candidate example words linked to lexemes",
                sql=(
                    "SELECT CASE WHEN e.lexeme_id IS NULL THEN 'unlinked' "
                    "WHEN lower(e.example_word) = l.normalized THEN 'linked_casefold_equal' "
                    "ELSE 'linked_other' END AS link, "
                    "COUNT(*) AS examples, COUNT(DISTINCT e.syllable_id) AS syllables "
                    "FROM sx_candidate_example e LEFT JOIN lexeme l USING(lexeme_id) GROUP BY 1 ORDER BY 1"
                ),
                priority=19,
            )
        )
    return probes


def unit_probe(text: str) -> Probe:
    return Probe(
        id=f"syll.unit:{text}", family="syll.unit", title=f"Position profile of syllable '{text}'",
        sql=(
            f"SELECT position, COUNT(*) AS tokens, COUNT(DISTINCT t.variant_id) AS variants FROM ({TOKENS}) t "
            "WHERE lower(t.text) = ? GROUP BY 1 ORDER BY 1"
        ),
        params=(text,), priority=30,
    )


def analyse_top_tokens(probe: Probe, outcome: dict, ctx: Context) -> None:
    for syllable, *_ in outcome["rows"][: ctx.fanout]:
        ctx.child(probe, unit_probe(syllable))


def analyse_unit(probe: Probe, outcome: dict, ctx: Context) -> None:
    text = probe.params[0]
    profile = {row[0]: row[1] for row in outcome["rows"]}
    total = sum(profile.values())
    if total:
        dominant = max(profile, key=profile.get)
        ctx.findings.append(
            Finding(
                "syllable_profile", text,
                f"'{text}': {total} tokens, mostly {dominant} ({profile[dominant] / total:.0%})", profile,
            )
        )
    ctx.child(
        probe,
        Probe(
            id=f"syll.examples:{text}", family="syll.examples", title=f"Example words containing syllable '{text}'",
            sql=(
                "SELECT v.spelling, ws.ordinal, v.source_record FROM sx_word_syllable ws "
                "JOIN sx_syllable s USING(syllable_id) JOIN sx_word_variant v USING(variant_id) "
                "WHERE lower(s.text) = ? ORDER BY v.spelling, ws.ordinal LIMIT 25"
            ),
            params=(text,), priority=35, store_limit=25,
        ),
    )
    ctx.child(
        probe,
        Probe(
            id=f"syll.next:{text}", family="syll.neighbours", title=f"Syllables that follow '{text}'",
            sql=(
                "SELECT lower(s2.text) AS next_syllable, COUNT(*) AS tokens FROM sx_word_syllable a "
                "JOIN sx_syllable s1 ON s1.syllable_id = a.syllable_id "
                "JOIN sx_word_syllable b ON b.variant_id = a.variant_id AND b.ordinal = a.ordinal + 1 "
                "JOIN sx_syllable s2 ON s2.syllable_id = b.syllable_id WHERE lower(s1.text) = ? "
                "GROUP BY 1 ORDER BY tokens DESC, next_syllable LIMIT 25"
            ),
            params=(text,), priority=32, store_limit=25,
        ),
    )


def analyse_neighbours(probe: Probe, outcome: dict, ctx: Context) -> None:
    for syllable, *_ in outcome["rows"][: ctx.fanout]:
        ctx.child(probe, unit_probe(syllable))


def analyse_boundary_bigrams(probe: Probe, outcome: dict, ctx: Context) -> None:
    for bigram, boundaries in outcome["rows"][: ctx.fanout]:
        ctx.child(
            probe,
            Probe(
                id=f"syll.bigram_split:{bigram}", family="syll.bigram_split",
                title=f"Letter pair '{bigram}': across a boundary vs inside one syllable",
                sql=(
                    "SELECT ? AS bigram, ? AS across_boundary, "
                    "(SELECT COUNT(*) FROM sx_word_syllable ws JOIN sx_syllable s USING(syllable_id) "
                    " WHERE instr(lower(s.text), ?) > 0) AS syllable_tokens_containing"
                ),
                params=(bigram, boundaries, bigram), priority=40,
            ),
        )


def analyse_bigram_split(probe: Probe, outcome: dict, ctx: Context) -> None:
    bigram, across, inside = outcome["rows"][0]
    total = across + inside
    if total:
        ctx.findings.append(
            Finding(
                "bigram_split", bigram,
                f"'{bigram}' split across a boundary {across} times; inside a syllable token "
                f"{inside} times ({across / total:.0%} split)",
                {"across": across, "inside_tokens": inside},
            )
        )


def analyse_count_vs_cmu(probe: Probe, outcome: dict, ctx: Context) -> None:
    rows = outcome["rows"]
    total = sum(r[2] for r in rows)
    agree = sum(r[2] for r in rows if r[0] == r[1])
    if total:
        ctx.findings.append(
            Finding(
                "agreement", "syllable_count_vs_cmu",
                f"Webster and CMU syllable counts agree for {agree}/{total} variants ({agree / total:.1%})",
                {"agree": agree, "total": total},
            )
        )
    for webster_n, cmu_n, _ in [r for r in rows if r[0] != r[1]][: ctx.fanout]:
        ctx.child(
            probe,
            Probe(
                id=f"syll.count_mismatch:{webster_n}:{cmu_n}", family="syll.examples",
                title=f"Examples: Webster {webster_n} syllables, CMU {cmu_n}",
                sql=(
                    "WITH w AS (SELECT v.variant_id, v.lexeme_id, v.spelling, v.source_record, COUNT(*) AS n "
                    "FROM sx_word_variant v JOIN sx_word_syllable ws USING(variant_id) GROUP BY v.variant_id), "
                    f"pp AS ({PRIMARY_PRON}) "
                    "SELECT w.spelling, w.source_record, p.phones, p.stress_pattern FROM w JOIN pp USING(lexeme_id) "
                    "JOIN pronunciation p ON p.pronunciation_id = pp.pronunciation_id "
                    "WHERE w.n = ? AND p.syllable_count = ? ORDER BY w.spelling LIMIT 25"
                ),
                params=(webster_n, cmu_n), priority=36, store_limit=25,
            ),
        )


def analyse_stress(probe: Probe, outcome: dict, ctx: Context) -> None:
    rows = outcome["rows"]
    compared = [r for r in rows if r[0] == "compare"]
    total = sum(r[4] for r in compared)
    agree = sum(r[4] for r in compared if r[2] == r[3])
    if total:
        ctx.findings.append(
            Finding(
                "agreement", "primary_stress_vs_cmu",
                f"Webster primary-stress syllable equals CMU primary stress for {agree}/{total} "
                f"comparable variants ({agree / total:.1%})",
                {"agree": agree, "total": total},
            )
        )
    disagreements = sorted((r for r in compared if r[2] != r[3]), key=lambda r: -r[4])[: ctx.fanout]
    for _, n, webster_idx, cmu_idx, _count in disagreements:
        ctx.child(
            probe,
            Probe(
                id=f"syll.stress_mismatch:{n}:{webster_idx}:{cmu_idx}", family="syll.examples",
                py="stress_examples",
                title=f"Examples: {n} syllables, Webster primary {webster_idx}, CMU primary {cmu_idx}",
                params=(n, webster_idx, cmu_idx), priority=37, store_limit=25,
            ),
        )


ANALYSERS: dict[str, Callable[[Probe, dict, Context], None]] = {
    "table.count": analyse_count,
    "column.profile": analyse_profile,
    "syll.top_tokens": analyse_top_tokens,
    "syll.unit": analyse_unit,
    "syll.neighbours": analyse_neighbours,
    "syll.boundary_bigrams": analyse_boundary_bigrams,
    "syll.bigram_split": analyse_bigram_split,
    "syll.count_vs_cmu": analyse_count_vs_cmu,
    "syll.stress_vs_cmu": analyse_stress,
}


# ---------------------------------------------------------------- python probes


def parse_record(record: str) -> list[tuple[str, str]]:
    """Split a Webster-style record body into (syllable, following mark) pairs."""
    body = record.split("=", 1)[1] if "=" in record else record
    parts: list[tuple[str, str]] = []
    current = ""
    for ch in body:
        if ch in STRESS_MARKS:
            if current:
                parts.append((current, STRESS_MARKS[ch]))
            current = ""
        else:
            current += ch
    if current:
        parts.append((current, "end"))
    return parts


def _stress_rows(con: sqlite3.Connection):
    spans = dict(con.execute("SELECT variant_id, COUNT(*) FROM sx_word_syllable GROUP BY 1"))
    cursor = con.execute(
        f"WITH pp AS ({PRIMARY_PRON}) SELECT v.variant_id, v.spelling, v.source_record, p.phones, "
        "p.stress_pattern, p.syllable_count FROM sx_word_variant v LEFT JOIN pp USING(lexeme_id) "
        "LEFT JOIN pronunciation p ON p.pronunciation_id = pp.pronunciation_id ORDER BY v.variant_id"
    )
    for variant_id, spelling, record, phones, stress, cmu_n in cursor:
        parts = parse_record(record)
        n = spans.get(variant_id, 0)
        example = (spelling, record, phones, stress)
        if len(parts) != n:
            yield "record_parts_differ_from_spans", n, None, None, example
            continue
        primaries = [i for i, (_, mark) in enumerate(parts) if mark == "primary"]
        webster_idx = primaries[0] if len(primaries) == 1 else (-1 if not primaries else -2)
        if stress is None:
            yield "no_cmu", n, webster_idx, None, example
        elif cmu_n != n or len(stress) != n:
            yield "syllable_count_differs", n, webster_idx, None, example
        elif webster_idx < 0 or "1" not in stress:
            yield "primary_absent_or_multiple", n, webster_idx, stress.find("1"), example
        else:
            yield "compare", n, webster_idx, stress.find("1"), example


def py_stress_vs_cmu(con: sqlite3.Connection, params: tuple):
    counts: Counter = Counter()
    for status, n, w, c, _ in _stress_rows(con):
        counts[(status, n, w, c)] += 1

    def order(r):
        return (r[0] != "compare", -r[4], r[0], r[1], -99 if r[2] is None else r[2], -99 if r[3] is None else r[3])

    rows = sorted(([*key, value] for key, value in counts.items()), key=order)
    return ["status", "syllables", "webster_primary_index", "cmu_primary_index", "variants"], rows


def py_stress_examples(con: sqlite3.Connection, params: tuple):
    n, webster_idx, cmu_idx = params
    rows = sorted(
        list(example)
        for status, sn, w, c, example in _stress_rows(con)
        if status == "compare" and sn == n and w == webster_idx and c == cmu_idx
    )
    return ["spelling", "source_record", "cmu_phones", "cmu_stress"], rows[:25]


def py_phonetic_structure(con: sqlite3.Connection, params: tuple):
    phones: dict[int, list[tuple[str, str]]] = {}
    for syllable_id, symbol, role in con.execute(
        "SELECT syllable_id, symbol, role FROM sx_phone ORDER BY syllable_id, ordinal"
    ):
        phones.setdefault(syllable_id, []).append((symbol, role))
    counters = {k: Counter() for k in ("text_check", "shape", "onset", "nucleus", "coda")}
    for syllable_id, text, status in con.execute(
        "SELECT syllable_id, text, analysis_status FROM sx_syllable WHERE kind='phonetic'"
    ):
        seq = phones.get(syllable_id, [])
        counters["text_check"]["equal" if " ".join(s for s, _ in seq) == text else "differs"] += 1
        if status != "source_reported":
            counters["shape"][f"[{status}]"] += 1
            continue
        onset = [s for s, r in seq if r == "onset"]
        coda = [s for s, r in seq if r == "coda"]
        nucleus = [s for s, r in seq if r == "nucleus"]
        counters["shape"]["C" * len(onset) + "V" * len(nucleus) + "C" * len(coda)] += 1
        counters["onset"][" ".join(onset) or "(none)"] += 1
        counters["nucleus"][" ".join(nucleus)] += 1
        counters["coda"][" ".join(coda) or "(none)"] += 1
    rows = []
    for facet, counter in counters.items():
        for value, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append([facet, value, count])
    return ["facet", "value", "syllable_types"], rows


def py_phone_inventory(con: sqlite3.Connection, params: tuple):
    beep = Counter(dict(con.execute("SELECT symbol, COUNT(*) FROM sx_phone GROUP BY 1")))
    cmu: Counter = Counter()
    for (phones,) in con.execute("SELECT phones FROM pronunciation"):
        for phone in phones.split("#", 1)[0].split():
            cmu[phone.rstrip("012").lower()] += 1
    workbook: set[str] = set()
    try:
        workbook = {
            row[0].lower()
            for row in con.execute("SELECT phone_symbol FROM sx_workbook_row_key WHERE phone_symbol IS NOT NULL")
        }
    except sqlite3.OperationalError:
        pass
    rows = [
        [symbol, beep.get(symbol, 0), cmu.get(symbol, 0), int(symbol in workbook)]
        for symbol in sorted(set(beep) | set(cmu) | workbook)
    ]
    return ["symbol", "beep_phone_rows", "cmu_phone_tokens", "in_workbook_phoneme_sheet"], rows


PY_PROBES: dict[str, Callable[[sqlite3.Connection, tuple], tuple[list[str], Iterable]]] = {
    "stress_vs_cmu": py_stress_vs_cmu,
    "stress_examples": py_stress_examples,
    "phonetic_structure": py_phonetic_structure,
    "phone_inventory": py_phone_inventory,
}


# ---------------------------------------------------------------- driver


def seed(con: sqlite3.Connection) -> list[Probe]:
    tables = {
        row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }
    return syllable_probes(tables) + schema_probes(con)


def explore(
    db: Path,
    ledger_path: Path,
    max_probes: int | None = None,
    budget_s: float | None = None,
    probe_timeout_s: float = 300.0,
    max_depth: int = 6,
    fanout: int = 10,
    watch: bool = False,
    poll_s: float = 60.0,
    log: Callable[[str], None] = print,
) -> int:
    db = Path(db)
    ledger = Ledger(ledger_path)
    src = open_source(db)
    state, new = ledger.state_for(db, src, log)
    added = ledger.add_probes(state, seed(src))
    log(f"state {state} ({'new' if new else 'resumed'}); seeded {added} new probes; ledger {ledger.path}")
    started = time.monotonic()
    executed = 0
    try:
        while True:
            if max_probes is not None and executed >= max_probes:
                break
            if budget_s is not None and time.monotonic() - started > budget_s:
                break
            probe = ledger.next_pending(state)
            if probe is None:
                if not watch:
                    break
                time.sleep(poll_s)
                src.close()
                src = open_source(db)
                next_state, new = ledger.state_for(db, src, log)
                if new:
                    state = next_state
                    log(f"source changed; new state {state}; seeded {ledger.add_probes(state, seed(src))} probes")
                continue
            outcome = run_probe(src, probe, probe_timeout_s)
            ctx = Context(max_depth=max_depth, fanout=fanout)
            analyse(probe, outcome, ctx)
            ledger.record(state, probe, outcome, ctx.findings)
            grown = ledger.add_probes(state, ctx.children)
            executed += 1
            counts = ledger.counts(state)
            done = sum(v for k, v in counts.items() if k != "pending")
            log(
                f"[{done}/{done + counts.get('pending', 0)}] {outcome['status']:7} {outcome['elapsed_s']:8.2f}s "
                f"{probe.id}" + (f" (+{grown} discovered)" if grown else "")
            )
    finally:
        src.close()
        ledger.close()
    return state


# ---------------------------------------------------------------- report


def _md_table(columns: list[str], rows: list[list[Any]], limit: int = 30) -> str:
    def cell(value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"\n_{len(rows) - limit} more stored rows in the ledger._")
    return "\n".join(lines)


def _decode_code(value_json: str) -> str:
    try:
        value = json.loads(value_json)
    except (TypeError, ValueError):
        return str(value_json)
    return str(value).strip() if value is not None else ""


def build_report(ledger_path: Path, state: int | None = None) -> str:
    con = sqlite3.connect(Path(ledger_path))
    state = state or con.execute("SELECT MAX(state_id) FROM source_state").fetchone()[0]
    db_path, size, sha, schema_sha, first_seen = con.execute(
        "SELECT db_path, size, sha256, schema_sha256, first_seen FROM source_state WHERE state_id=?", (state,)
    ).fetchone()
    results = {
        pid: (json.loads(cols) if cols else None, json.loads(rows) if rows else None, count, digest)
        for pid, cols, rows, count, digest in con.execute(
            "SELECT probe_id, columns_json, rows_json, row_count, result_sha256 FROM result WHERE state_id=?",
            (state,),
        )
    }
    titles = dict(con.execute("SELECT probe_id, title FROM probe WHERE state_id=?", (state,)))
    status = dict(con.execute("SELECT status, COUNT(*) FROM probe WHERE state_id=? GROUP BY 1", (state,)))
    by_kind: dict[str, list] = {}
    for kind, subject, summary, pid in con.execute(
        "SELECT kind, subject, summary, probe_id FROM finding WHERE state_id=? ORDER BY kind, subject", (state,)
    ):
        by_kind.setdefault(kind, []).append((subject, summary, pid))
    con.close()

    out = [
        "# Lexicon exploration report",
        "",
        f"- Source: `{db_path}` ({size:,} bytes)",
        f"- Source SHA-256: `{sha}`",
        f"- Schema SHA-256: `{schema_sha}`",
        f"- State {state}, first seen {first_seen}",
        "- Probes: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())),
        "",
    ]

    def section(pid: str, heading: str | None = None, limit: int = 30) -> None:
        if pid not in results or results[pid][0] is None:
            return
        cols, rows, count, digest = results[pid]
        out.extend([
            f"### {heading or titles.get(pid, pid)}", "", f"`{pid}` rows={count} sha256=`{digest}`", "",
            _md_table(cols, rows or [], limit), "",
        ])

    out += ["## Integrity invariants", ""]
    inv = [(s, "PASS", sm) for s, sm, _ in by_kind.get("invariant_pass", [])]
    inv += [(s, "FAIL", sm) for s, sm, _ in by_kind.get("invariant_fail", [])]
    out += [_md_table(["probe", "result", "detail"], sorted(inv, key=lambda r: (r[1], r[0])), limit=1000), ""]
    for subject, _, _ in by_kind.get("invariant_fail", []):
        section(f"sample:{subject}", f"Violations sample: {subject}", limit=10)

    out += ["## Syllable layer", ""]
    out += [f"- {summary}" for _, summary, _ in by_kind.get("agreement", [])] + [""]
    for pid in ("syll.inventory", "syll.coverage", "syll.per_word", "syll.case_types", "syll.multi_variant"):
        section(pid, limit=20)
    section("syll.top_tokens", limit=40)
    section("syll.length_by_position", limit=60)
    section("syll.boundary_bigrams", limit=30)
    bigrams = by_kind.get("bigram_split", [])
    if bigrams:
        out += ["### Boundary letter pairs: split vs inside", ""] + [f"- {s}" for _, s, _ in bigrams] + [""]
    section("syll.count_vs_cmu", limit=25)
    for pid in sorted(p for p in results if p.startswith("syll.count_mismatch:")):
        section(pid, limit=8)
    section("syll.stress_vs_cmu", limit=40)
    for pid in sorted(p for p in results if p.startswith("syll.stress_mismatch:")):
        section(pid, limit=8)
    section("phon.candidate_links")
    if results.get("phon.structure", (None, None))[1]:
        cols, rows, count, digest = results["phon.structure"]
        out += [f"### {titles['phon.structure']}", "", f"`phon.structure` rows={count} sha256=`{digest}`", ""]
        for facet in ("text_check", "shape", "onset", "nucleus", "coda"):
            out += [f"**{facet}**", "", _md_table(cols, [r for r in rows if r[0] == facet], 15), ""]
    section("phon.inventory_vs_cmu", limit=100)

    if results.get("wb.section_codes_by_slot", (None, None))[1]:
        cols, rows, count, digest = results["wb.section_codes_by_slot"]
        totals: dict[str, Counter] = {}
        for sec, _slot, value_json, tokens, _attrs in rows:
            totals.setdefault(sec, Counter())[_decode_code(value_json)] += tokens
        table = []
        for sec, counter in sorted(totals.items()):
            total = sum(counter.values())
            resolved = sum(v for k, v in counter.items() if k not in UNRESOLVED and k != "0")
            top = ", ".join(f"{k}:{v / total:.0%}" for k, v in counter.most_common(4))
            table.append([sec, total, f"{resolved / total:.1%}", top])
        out += [
            "## Canonical 437 workbook through the syllable layer", "",
            "Cell codes per section, weighted by letter tokens inside Webster syllables "
            "(resolved = not in the conical `UNRESOLVED` set and not `0`).", "",
            f"`wb.section_codes_by_slot` rows={count} sha256=`{digest}`", "",
            _md_table(["section", "attribute x letter tokens", "resolved", "top codes"], table, 100), "",
        ]

    profiles = by_kind.get("syllable_profile", [])
    if profiles:
        out += [
            "## Discovery walk (syllable graph)", "",
            f"{len(profiles)} syllable units explored by following the most frequent successors:", "",
        ] + [f"- {s}" for _, s, _ in profiles[:300]] + [""]

    out += ["## Whole-lexicon schema findings", ""]
    for kind, label in (
        ("empty_table", "Empty tables"), ("all_null", "All-NULL columns"), ("constant", "Constant columns"),
        ("partial_null", "Partially NULL columns"), ("timeout", "Probes that timed out"),
        ("error", "Probes that errored"),
    ):
        items = by_kind.get(kind, [])
        if items:
            out += [f"### {label} ({len(items)})", ""] + [f"- {s}" for _, s, _ in items[:300]] + [""]

    out += [
        "## Honest boundaries", "",
        "- Read-only: the source is opened with `mode=ro` and `PRAGMA query_only`; all output goes to the ledger.",
        "- Every number is a direct count over the source; nothing is imputed. Workbook codes stay symbolic.",
        "- Webster stress is read from the source record convention (mark after the stressed syllable: "
        "`\"` primary, `` ` `` secondary, `*` plain break). CMU stress is its first pronunciation per lexeme; "
        "stress is only compared when both sources have the same syllable count.",
        "- Letter slots (first/interior/last/only) are orthographic positions inside Webster syllables, "
        "not phonological onset/coda.",
        "- BEEP phonetic syllables are candidate membership with example words, not letter-aligned spans.",
        "- Result SHA-256 covers every row of a probe; only the first rows are stored, so long tables are truncated here.",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="explore (resumable; rerun the same command to continue)")
    run.add_argument("--db", required=True, type=Path)
    run.add_argument("--ledger", required=True, type=Path)
    run.add_argument("--max-probes", type=int)
    run.add_argument("--budget-seconds", type=float)
    run.add_argument("--probe-timeout", type=float, default=300.0)
    run.add_argument("--max-depth", type=int, default=6)
    run.add_argument("--fanout", type=int, default=10)
    run.add_argument("--watch", action="store_true", help="keep running; re-explore when the source changes")
    run.add_argument("--poll-seconds", type=float, default=60.0)
    run.add_argument("--report", type=Path, help="write a Markdown report when the run stops")
    rep = sub.add_parser("report", help="write a Markdown report from a ledger")
    rep.add_argument("--ledger", required=True, type=Path)
    rep.add_argument("--out", required=True, type=Path)
    rep.add_argument("--state", type=int)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    if args.command == "run":
        state = None
        try:
            state = explore(
                args.db, args.ledger, args.max_probes, args.budget_seconds, args.probe_timeout,
                args.max_depth, args.fanout, args.watch, args.poll_seconds,
            )
        except KeyboardInterrupt:
            print("stopped; rerun the same command to resume")
        if args.report:
            args.report.write_text(build_report(args.ledger, state), encoding="utf-8")
            print(f"report: {args.report}")
    else:
        args.out.write_text(build_report(args.ledger, args.state), encoding="utf-8")
        print(f"report: {args.out}")


if __name__ == "__main__":
    main()
