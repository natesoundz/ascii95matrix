#!/usr/bin/env python3
"""Project warranted real evidence into the scalar g[i,k] interface.

This is intentionally strict. It fills only measurements whose numeric meaning
is explicit and deterministic. Typed/category/relation fields without a
declared scalar projection remain NaN and are reported, never coerced to zero.

The resulting shard format matches relational_compiler.py:
  y         int16   [N]      observed ASCII code
  g         float64 [N,437]  valid scalar or NaN
  w         float64 [N]      evidence weight
  source_id str     [N]      occurrence trace id

A strict full-437 build fails if any eligible dimension has zero warranted
numeric coverage. --allow-incomplete permits writing diagnostic shards but does
not make them valid full-437 compiler input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import math
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np


def _fmt_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _ram_gb() -> float:
    """Current process RSS in GiB; Windows path needs no third-party package."""
    try:
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return float(pmc.WorkingSetSize) / (1024.0 ** 3)
    except Exception:
        pass
    return math.nan


def _vram_gb() -> float:
    """Report this process's NVIDIA compute VRAM when nvidia-smi is available."""
    try:
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if cp.returncode != 0:
            return math.nan
        pid = str(__import__("os").getpid())
        mib = 0.0
        found = False
        for line in cp.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0] == pid:
                mib += float(parts[1])
                found = True
        return mib / 1024.0 if found else 0.0
    except Exception:
        return math.nan


def _resource_text() -> str:
    ram = _ram_gb()
    vram = _vram_gb()
    rs = "N/A" if not math.isfinite(ram) else f"{ram:.2f} GB"
    vs = "N/A" if not math.isfinite(vram) else f"{vram:.2f} GB"
    return f"RAM {rs} | VRAM {vs}"


def _progress(stage: str, done: int, total: int, started: float) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    pct = 100.0 * done / total if total else 0.0
    remaining = (
        elapsed * (total - done) / done
        if done > 0 and total > done
        else 0.0 if done >= total else math.nan
    )
    print(
        f"{stage} | {done:,}/{total:,} | {pct:6.2f}% | "
        f"ELAPSED {_fmt_duration(elapsed)} | REMAINING {_fmt_duration(remaining)} | "
        f"{_resource_text()}",
        flush=True,
    )

ASCII_MIN = 32
ASCII_MAX = 126
V = 95
D = 437


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield obj


def word_span(text: str, position: int) -> tuple[int, int]:
    if not text[position].isalnum() and text[position] not in "_'":
        return position, position + 1
    left = position
    right = position + 1
    while left > 0 and (text[left - 1].isalnum() or text[left - 1] in "_'"):
        left -= 1
    while right < len(text) and (text[right].isalnum() or text[right] in "_'"):
        right += 1
    return left, right


def repetition_count(text: str, position: int) -> int:
    ch = text[position]
    left = position
    right = position + 1
    while left > 0 and text[left - 1] == ch:
        left -= 1
    while right < len(text) and text[right] == ch:
        right += 1
    return right - left


class CorpusStats:
    def __init__(self) -> None:
        self.unigram = np.zeros(V, dtype=np.int64)
        self.bigram = np.zeros((V, V), dtype=np.int64)
        self.trigram = defaultdict(Counter)
        self.initial_word = np.zeros(V, dtype=np.int64)
        self.medial_word = np.zeros(V, dtype=np.int64)
        self.final_word = np.zeros(V, dtype=np.int64)
        self.initial_sentence = np.zeros(V, dtype=np.int64)
        self.final_sentence = np.zeros(V, dtype=np.int64)
        self.total = 0

    def add(self, text: str) -> None:
        ids = np.fromiter((ord(c) - ASCII_MIN for c in text), dtype=np.int16)
        self.total += len(ids)
        if len(ids):
            self.initial_sentence[int(ids[0])] += 1
            self.final_sentence[int(ids[-1])] += 1
        for i, c in enumerate(ids.tolist()):
            self.unigram[c] += 1
            if i:
                self.bigram[int(ids[i - 1]), c] += 1
            if i >= 2:
                self.trigram[(int(ids[i - 2]), int(ids[i - 1]))][c] += 1

        for m in re.finditer(r"[A-Za-z0-9_']+", text):
            a, b = m.span()
            if a < b:
                self.initial_word[ord(text[a]) - ASCII_MIN] += 1
                self.final_word[ord(text[b - 1]) - ASCII_MIN] += 1
                for p in range(a + 1, b - 1):
                    self.medial_word[ord(text[p]) - ASCII_MIN] += 1

    def next_entropy(self, c: int) -> float:
        row = self.bigram[c].astype(np.float64)
        total = float(row.sum())
        if total <= 0.0:
            return math.nan
        p = row[row > 0] / total
        return float(-(p * np.log2(p)).sum())

    def previous_entropy(self, c: int) -> float:
        col = self.bigram[:, c].astype(np.float64)
        total = float(col.sum())
        if total <= 0.0:
            return math.nan
        p = col[col > 0] / total
        return float(-(p * np.log2(p)).sum())


def validate_record(record: dict) -> tuple[str, list[int]]:
    text = record.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("record text must be nonempty string")
    if any(not ASCII_MIN <= ord(c) <= ASCII_MAX for c in text):
        raise ValueError("record text is not literal printable ASCII95")
    positions = record.get("ground_truth_positions")
    if positions is None:
        positions = list(range(len(text)))
    if (
        not isinstance(positions, list)
        or any(type(x) is not int for x in positions)
        or positions != sorted(set(positions))
        or any(x < 0 or x >= len(text) for x in positions)
    ):
        raise ValueError("invalid ground_truth_positions")
    return text, positions


def corpus_pass(records: Path) -> tuple[CorpusStats, int, int]:
    stats = CorpusStats()
    record_count = 0
    truth_count = 0
    for record in iter_jsonl(records):
        text, positions = validate_record(record)
        stats.add(text)
        record_count += 1
        truth_count += len(positions)
    if truth_count == 0:
        raise ValueError("no ground-truth positions")
    return stats, record_count, truth_count


def index_dimensions(registry: dict) -> tuple[list[dict], dict[str, int]]:
    dims = registry["dimensions"]
    if len(dims) != D:
        raise ValueError(f"expected {D} registry dimensions")
    by_id = {d["id"]: int(d["index"]) for d in dims}
    if len(by_id) != D:
        raise ValueError("dimension IDs are not unique")
    return dims, by_id


def set_if_present(g: np.ndarray, by_id: dict[str, int], key: str, value: float) -> None:
    k = by_id.get(key)
    if k is not None and math.isfinite(value):
        g[k] = value


def apply_fixed_status(g: np.ndarray, dimensions: list[dict], char_index: int) -> None:
    for d in dimensions:
        status = d["status_by_ascii"][char_index]
        k = int(d["index"])
        if status == "1":
            g[k] = 1.0
        elif status == "0":
            g[k] = 0.0
        # X, K and all deferred/typed statuses deliberately remain NaN.


def apply_numeric_facts(g: np.ndarray, by_id: dict[str, int], ch: str) -> None:
    if not ch.isdigit():
        return
    value = float(ord(ch) - ord("0"))
    for key in (
        "character_identity.numeric_value",
        "numeric.decimal.numeric_value",
        "numeric.decimal.integer",
        "numeric.decimal.digit_rank",
    ):
        set_if_present(g, by_id, key, value)


def apply_orthographic(
    g: np.ndarray,
    by_id: dict[str, int],
    text: str,
    p: int,
) -> None:
    n = len(text)
    a, b = word_span(text, p)
    wl = b - a
    wp = p - a

    set_if_present(g, by_id, "orthographic.absolute_character_position", float(p))
    set_if_present(
        g,
        by_id,
        "orthographic.relative_word_position",
        0.0 if wl <= 1 else float(wp / (wl - 1)),
    )
    set_if_present(
        g,
        by_id,
        "orthographic.relative_sentence_position",
        0.0 if n <= 1 else float(p / (n - 1)),
    )
    set_if_present(g, by_id, "orthographic.word_initial", float(p == a))
    set_if_present(g, by_id, "orthographic.word_medial", float(a < p < b - 1))
    set_if_present(g, by_id, "orthographic.word_final", float(p == b - 1))
    set_if_present(g, by_id, "orthographic.sentence_initial", float(p == 0))
    set_if_present(g, by_id, "orthographic.sentence_medial", float(0 < p < n - 1))
    set_if_present(g, by_id, "orthographic.sentence_final", float(p == n - 1))
    # records.jsonl currently stores one physical line per local-file record.
    set_if_present(g, by_id, "orthographic.line_initial", float(p == 0))
    set_if_present(g, by_id, "orthographic.line_final", float(p == n - 1))
    set_if_present(g, by_id, "orthographic.preceded_by_space", float(p > 0 and text[p - 1] == " "))
    set_if_present(g, by_id, "orthographic.followed_by_space", float(p + 1 < n and text[p + 1] == " "))
    set_if_present(g, by_id, "orthographic.distance_to_left_boundary", float(wp))
    set_if_present(g, by_id, "orthographic.distance_to_right_boundary", float(b - p - 1))
    set_if_present(g, by_id, "orthographic.word_length", float(wl))
    rc = repetition_count(text, p)
    set_if_present(g, by_id, "orthographic.repeated_character_status", float(rc > 1))
    set_if_present(g, by_id, "orthographic.character_repetition_count", float(rc))


def apply_corpus_stats(
    g: np.ndarray,
    by_id: dict[str, int],
    stats: CorpusStats,
    text: str,
    p: int,
) -> None:
    c = ord(text[p]) - ASCII_MIN
    total = max(1, stats.total)
    set_if_present(
        g,
        by_id,
        "corpus.unigram_frequency",
        float(stats.unigram[c] / total),
    )

    ent = stats.next_entropy(c)
    if math.isfinite(ent):
        set_if_present(g, by_id, "corpus.next_character_entropy", ent)

    # The registry's conditional entropy is context-dependent. Use the
    # previous-character distribution of the current target as the warranted
    # reverse conditional entropy, and preserve this declared method in report.
    prev_ent = stats.previous_entropy(c)
    if math.isfinite(prev_ent):
        set_if_present(g, by_id, "corpus.conditional_entropy", prev_ent)

    denom = max(1, int(stats.unigram[c]))
    set_if_present(
        g,
        by_id,
        "corpus.word_initial_frequency",
        float(stats.initial_word[c] / denom),
    )
    set_if_present(
        g,
        by_id,
        "corpus.word_medial_frequency",
        float(stats.medial_word[c] / denom),
    )
    set_if_present(
        g,
        by_id,
        "corpus.word_final_frequency",
        float(stats.final_word[c] / denom),
    )
    set_if_present(
        g,
        by_id,
        "corpus.sentence_initial_frequency",
        float(stats.initial_sentence[c] / denom),
    )
    set_if_present(
        g,
        by_id,
        "corpus.sentence_final_frequency",
        float(stats.final_sentence[c] / denom),
    )


def apply_observed_neighbor_strengths(
    g: np.ndarray,
    by_id: dict[str, int],
    stats: CorpusStats,
    text: str,
    p: int,
) -> None:
    c = ord(text[p]) - ASCII_MIN
    if p + 1 < len(text):
        r = ord(text[p + 1]) - ASCII_MIN
        row_total = float(stats.bigram[c].sum())
        if row_total > 0:
            set_if_present(
                g,
                by_id,
                "corpus.next_character_probability",
                float(stats.bigram[c, r] / row_total),
            )
    if p > 0:
        left = ord(text[p - 1]) - ASCII_MIN
        col_total = float(stats.bigram[:, c].sum())
        if col_total > 0:
            set_if_present(
                g,
                by_id,
                "corpus.previous_character_probability",
                float(stats.bigram[left, c] / col_total),
            )


def apply_evidence_statistics(
    g: np.ndarray,
    by_id: dict[str, int],
    stats: CorpusStats,
    c: int,
) -> None:
    obs = float(stats.unigram[c])
    total = float(max(1, stats.total))
    set_if_present(g, by_id, "evidence.observation_count", obs)
    set_if_present(g, by_id, "evidence.support_count", obs)
    set_if_present(g, by_id, "evidence.contradiction_count", 0.0)
    set_if_present(g, by_id, "evidence.sample_size", total)
    set_if_present(g, by_id, "evidence.coverage", float(obs > 0.0))
    set_if_present(g, by_id, "evidence.occurrence_rate", obs / total)
    set_if_present(g, by_id, "source_class.corpus_derived_statistic", 1.0)
    set_if_present(
        g,
        by_id,
        "source_class.deterministic_extraction_from_published_raw_data",
        1.0,
    )



def normalize_lexeme(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().replace("_", " ")).casefold()


def load_lexical_snapshots(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    result: dict[str, dict] = {}
    for item in iter_jsonl(path):
        normalized = str(item.get("normalized", "")).strip()
        if normalized:
            result[normalized] = item
    return result


def containing_db_token(record: dict, position: int) -> dict | None:
    annotations = record.get("annotations")
    if not isinstance(annotations, dict):
        return None
    tokens = annotations.get("tokens")
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token, dict):
            continue
        a = token.get("char_start")
        b = token.get("char_end")
        if type(a) is int and type(b) is int and a <= position < b:
            return token
    return None


def containing_python_token(record: dict, position: int) -> dict | None:
    if record.get("objective") != "CODE_CAUSAL":
        return None
    annotations = record.get("annotations")
    if not isinstance(annotations, dict):
        return None
    tokens = annotations.get("python_tokens")
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token, dict):
            continue
        a = token.get("start_col")
        b = token.get("end_col")
        if type(a) is int and type(b) is int and a <= position < b:
            return token
    return None


def apply_grammar_from_ud(
    g: np.ndarray,
    by_id: dict[str, int],
    record: dict,
    position: int,
) -> None:
    """Project binary grammatical participation from the real aligned UD token.

    Categorical POS/dependency/head identities remain unresolved because the
    scalar contract has no declared lossless categorical encoding.
    """
    token = containing_db_token(record, position)
    if token is None:
        return
    dep = str(token.get("dep", "") or "").casefold()
    if not dep:
        return
    base = dep.split(":", 1)[0]
    roles = {
        "grammar.subject_participation": base in {"nsubj", "csubj"},
        "grammar.object_participation": base in {"obj", "iobj"},
        "grammar.predicate_participation": base == "root",
        "grammar.modifier_participation": base in {
            "amod", "advmod", "nmod", "acl", "advcl", "appos", "nummod"
        },
        "grammar.determiner_participation": base == "det",
        "grammar.auxiliary_participation": base in {"aux", "cop"},
    }
    for key, active in roles.items():
        set_if_present(g, by_id, key, float(active))


def apply_semantic_from_lexical_snapshot(
    g: np.ndarray,
    by_id: dict[str, int],
    record: dict,
    position: int,
    lexical: dict[str, dict],
) -> None:
    """Project only source-backed binary lexical/relation participation."""
    token = containing_db_token(record, position)
    if token is None:
        return
    normalized = normalize_lexeme(str(token.get("lemma", "") or token.get("text", "")))
    snap = lexical.get(normalized)
    if not snap or not isinstance(snap.get("lexeme"), dict):
        return
    facts = snap.get("facts")
    if not isinstance(facts, dict):
        facts = {}

    set_if_present(g, by_id, "semantic.lexeme_membership", 1.0)
    senses = facts.get("senses")
    if isinstance(senses, list):
        set_if_present(g, by_id, "semantic.sense_membership", float(bool(senses)))

    relations = facts.get("relations")
    if not isinstance(relations, list):
        return
    relation_types = {
        str(row.get("relation_type", "")).upper()
        for row in relations
        if isinstance(row, dict) and row.get("relation_type")
    }
    relation_laws = {
        "semantic.antonym_relation": {"ANTONYM"},
        "semantic.hypernym_relation": {"HYPERNYM"},
        "semantic.hyponym_relation": {"HYPONYM"},
        "semantic.meronym_relation": {
            "MEMBER_MERONYM", "PART_MERONYM", "SUBSTANCE_MERONYM"
        },
        "semantic.holonym_relation": {
            "MEMBER_HOLONYM", "PART_HOLONYM", "SUBSTANCE_HOLONYM"
        },
        "semantic.derivational_relation": {"DERIVATIONALLY_RELATED"},
        "semantic.similar_to_relation": {"SIMILAR_TO"},
    }
    for key, accepted in relation_laws.items():
        set_if_present(g, by_id, key, float(bool(relation_types & accepted)))


def apply_programming_from_python(
    g: np.ndarray,
    by_id: dict[str, int],
    record: dict,
    text: str,
    position: int,
) -> None:
    """Project Python roles only where the real tokenizer identifies a token."""
    token = containing_python_token(record, position)
    if token is None:
        return
    token_type = str(token.get("type", ""))
    token_text = str(token.get("text", ""))
    ch = text[position]
    is_name = token_type == "NAME" and not keyword.iskeyword(token_text)
    is_op = token_type == "OP"

    assignment = {"=", ":=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", ">>=", "<<="}
    comparison = {"==", "!=", "<", "<=", ">", ">=", "is", "in"}
    arithmetic = {"+", "-", "*", "/", "//", "%", "**", "@"}
    logical = {"and", "or", "not"}
    bitwise = {"&", "|", "^", "~", "<<", ">>"}
    delimiters = {"(", ")", "[", "]", "{", "}", ",", ":", ";"}

    roles = {
        "programming.identifier_participation": is_name,
        "programming.operator_participation": is_op or token_text in logical,
        "programming.assignment_operator": token_text in assignment,
        "programming.comparison_operator": token_text in comparison,
        "programming.arithmetic_operator": token_text in arithmetic,
        "programming.logical_operator": token_text in logical,
        "programming.bitwise_operator": token_text in bitwise,
        "programming.delimiter": token_text in delimiters,
        "programming.scope_opener": token_text in {"(", "[", "{"},
        "programming.scope_closer": token_text in {")", "]", "}"},
        "programming.statement_terminator": token_text == ";",
        "programming.comment_marker": token_type == "COMMENT" and ch == "#",
        "programming.string_delimiter": token_type == "STRING" and ch in {"'", '"'},
        "programming.escape_marker": token_type == "STRING" and ch == "\\",
        "programming.member_access_marker": token_text == ".",
        "programming.decorator_marker": token_text == "@",
    }
    for key, active in roles.items():
        set_if_present(g, by_id, key, float(active))


def source_id_for(record: dict, position: int) -> str:
    return f"{record.get('dataset','')}:{record.get('id','')}:{position}"


def write_shard(
    out_dir: Path,
    shard_index: int,
    y: list[int],
    g: list[np.ndarray],
    w: list[float],
    source_ids: list[str],
) -> Path:
    path = out_dir / f"evidence_{shard_index:06d}.npz"
    np.savez_compressed(
        path,
        y=np.asarray(y, dtype=np.int16),
        g=np.asarray(g, dtype=np.float64),
        w=np.asarray(w, dtype=np.float64),
        source_id=np.asarray(source_ids, dtype=str),
    )
    return path



def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "projection_checkpoint.json"


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _accumulate_saved_shard(
    path: Path,
    valid_counts: np.ndarray,
    char_valid: np.ndarray,
    min_value: np.ndarray,
    max_value: np.ndarray,
) -> int:
    with np.load(path, allow_pickle=False) as z:
        y = z["y"]
        g = z["g"]
        w = z["w"]
        source_id = z["source_id"]
        if g.ndim != 2 or g.shape[1] != D:
            raise ValueError(f"{path}: invalid g shape {g.shape}")
        n = int(g.shape[0])
        if y.shape != (n,) or w.shape != (n,) or source_id.shape != (n,):
            raise ValueError(f"{path}: inconsistent shard array lengths")
        if np.any((y < ASCII_MIN) | (y > ASCII_MAX)):
            raise ValueError(f"{path}: y outside ASCII95")
        valid = np.isfinite(g)
        valid_counts += valid.sum(axis=0, dtype=np.int64)
        for c in range(V):
            mask = y == (c + ASCII_MIN)
            if np.any(mask):
                char_valid[c] += valid[mask].sum(axis=0, dtype=np.int64)
        for k in range(D):
            vals = g[:, k]
            finite = vals[np.isfinite(vals)]
            if finite.size:
                min_value[k] = min(min_value[k], float(finite.min()))
                max_value[k] = max(max_value[k], float(finite.max()))
        return n


def _load_checkpoint(
    out_dir: Path,
    records_sha: str,
    registry_sha: str,
    lexical_sha: str | None,
    shard_size: int,
    truth_count: int,
    valid_counts: np.ndarray,
    char_valid: np.ndarray,
    min_value: np.ndarray,
    max_value: np.ndarray,
) -> tuple[list[dict], int, int]:
    path = _checkpoint_path(out_dir)
    if not path.exists():
        return [], 0, 0
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format": "ascii95-projection-checkpoint-v1",
        "records_sha256": records_sha,
        "registry_sha256": registry_sha,
        "lexical_snapshot_sha256": lexical_sha,
        "shard_size": shard_size,
        "ground_truth_occurrences": truth_count,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(
                f"checkpoint mismatch for {key}: saved={checkpoint.get(key)!r} current={value!r}; "
                "use a new --out directory or remove the incompatible checkpoint"
            )
    shards = checkpoint.get("completed_shards")
    if not isinstance(shards, list):
        raise RuntimeError("checkpoint completed_shards is not a list")
    expected_start = 0
    validated: list[dict] = []
    for index, item in enumerate(shards):
        if not isinstance(item, dict):
            raise RuntimeError("checkpoint shard entry is not an object")
        if int(item.get("index", -1)) != index:
            raise RuntimeError("checkpoint shard indexes are not contiguous")
        start = int(item.get("start_occurrence", -1))
        end = int(item.get("end_occurrence_exclusive", -1))
        if start != expected_start or end <= start:
            raise RuntimeError("checkpoint occurrence ranges are not contiguous")
        shard_path = out_dir / str(item.get("path", ""))
        if not shard_path.exists():
            raise RuntimeError(f"checkpoint shard missing: {shard_path}")
        actual_sha = sha256_file(shard_path)
        if actual_sha != item.get("sha256"):
            raise RuntimeError(f"checkpoint shard hash mismatch: {shard_path}")
        n = _accumulate_saved_shard(
            shard_path, valid_counts, char_valid, min_value, max_value
        )
        if n != end - start or n != int(item.get("occurrences", -1)):
            raise RuntimeError(f"checkpoint shard count mismatch: {shard_path}")
        validated.append(item)
        expected_start = end
    if expected_start > truth_count:
        raise RuntimeError("checkpoint exceeds current real occurrence count")
    return validated, expected_start, len(validated)


def _save_checkpoint(
    out_dir: Path,
    records_sha: str,
    registry_sha: str,
    lexical_sha: str | None,
    shard_size: int,
    truth_count: int,
    shards: list[dict],
) -> None:
    _atomic_json(
        _checkpoint_path(out_dir),
        {
            "format": "ascii95-projection-checkpoint-v1",
            "records_sha256": records_sha,
            "registry_sha256": registry_sha,
            "lexical_snapshot_sha256": lexical_sha,
            "shard_size": shard_size,
            "ground_truth_occurrences": truth_count,
            "completed_occurrences": (
                0 if not shards else int(shards[-1]["end_occurrence_exclusive"])
            ),
            "completed_shards": shards,
        },
    )


def build(
    records: Path,
    registry_path: Path,
    out_dir: Path,
    shard_size: int,
    allow_incomplete: bool,
    lexical_snapshot: Path | None = None,
) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    dimensions, by_id = index_dimensions(registry)
    prepass_started = time.monotonic()
    print(f"PRECOUNT START | {_resource_text()}", flush=True)
    stats, record_count, truth_count = corpus_pass(records)
    print(
        f"PRECOUNT DONE | RECORDS {record_count:,} | OCCURRENCES {truth_count:,} | "
        f"ELAPSED {_fmt_duration(time.monotonic() - prepass_started)} | {_resource_text()}",
        flush=True,
    )
    lexical = load_lexical_snapshots(lexical_snapshot)
    records_sha = sha256_file(records)
    registry_sha = sha256_file(registry_path)
    lexical_sha = lexical_sha

    out_dir.mkdir(parents=True, exist_ok=True)
    valid_counts = np.zeros(D, dtype=np.int64)
    char_valid = np.zeros((V, D), dtype=np.int64)
    min_value = np.full(D, np.inf, dtype=np.float64)
    max_value = np.full(D, -np.inf, dtype=np.float64)

    shards, resume_occurrence, shard_index = _load_checkpoint(
        out_dir,
        records_sha,
        registry_sha,
        lexical_sha,
        shard_size,
        truth_count,
        valid_counts,
        char_valid,
        min_value,
        max_value,
    )
    projection_started = time.monotonic()
    if resume_occurrence:
        print(
            f"CHECKPOINT RESUME | {resume_occurrence:,}/{truth_count:,} | "
            f"SHARDS {len(shards):,} | {_resource_text()}",
            flush=True,
        )
    else:
        stale = sorted(out_dir.glob("evidence_*.npz"))
        if stale:
            raise RuntimeError(
                "evidence shards exist without a matching checkpoint; "
                "use a new --out directory or remove the stale shards"
            )
        print(f"CHECKPOINT NEW | 0/{truth_count:,} | {_resource_text()}", flush=True)
    _progress("PROJECTION", resume_occurrence, truth_count, projection_started)

    y_buf: list[int] = []
    g_buf: list[np.ndarray] = []
    w_buf: list[float] = []
    source_buf: list[str] = []
    occurrences = resume_occurrence
    stream_occurrence = 0
    shard_start = resume_occurrence
    last_progress_time = projection_started
    progress_interval_seconds = 1.0

    def flush() -> None:
        nonlocal shard_index, shard_start
        if not y_buf:
            return
        path = write_shard(out_dir, shard_index, y_buf, g_buf, w_buf, source_buf)
        end_occurrence = shard_start + len(y_buf)
        shards.append(
            {
                "index": shard_index,
                "path": path.name,
                "sha256": sha256_file(path),
                "occurrences": len(y_buf),
                "start_occurrence": shard_start,
                "end_occurrence_exclusive": end_occurrence,
            }
        )
        _save_checkpoint(
            out_dir,
            records_sha,
            registry_sha,
            lexical_sha,
            shard_size,
            truth_count,
            shards,
        )
        print(
            f"CHECKPOINT SAVED | {end_occurrence:,}/{truth_count:,} | "
            f"SHARD {shard_index:06d} | {_resource_text()}",
            flush=True,
        )
        shard_index += 1
        shard_start = end_occurrence
        y_buf.clear()
        g_buf.clear()
        w_buf.clear()
        source_buf.clear()

    for record in iter_jsonl(records):
        text, positions = validate_record(record)
        for p in positions:
            if stream_occurrence < resume_occurrence:
                stream_occurrence += 1
                continue
            stream_occurrence += 1
            code = ord(text[p])
            c = code - ASCII_MIN
            row = np.full(D, np.nan, dtype=np.float64)
            apply_fixed_status(row, dimensions, c)
            apply_numeric_facts(row, by_id, text[p])
            apply_orthographic(row, by_id, text, p)
            apply_corpus_stats(row, by_id, stats, text, p)
            apply_observed_neighbor_strengths(row, by_id, stats, text, p)
            apply_evidence_statistics(row, by_id, stats, c)
            apply_grammar_from_ud(row, by_id, record, p)
            apply_semantic_from_lexical_snapshot(row, by_id, record, p, lexical)
            apply_programming_from_python(row, by_id, record, text, p)

            # Enforce canonical X/K boundaries after every resolver.
            for d in dimensions:
                status = d["status_by_ascii"][c]
                if status in ("X", "K"):
                    row[int(d["index"])] = np.nan

            valid = np.isfinite(row)
            valid_counts[valid] += 1
            char_valid[c, valid] += 1
            if np.any(valid):
                min_value[valid] = np.minimum(min_value[valid], row[valid])
                max_value[valid] = np.maximum(max_value[valid], row[valid])

            y_buf.append(code)
            g_buf.append(row)
            w_buf.append(1.0)
            source_buf.append(source_id_for(record, p))
            occurrences += 1
            now = time.monotonic()
            if now - last_progress_time >= progress_interval_seconds:
                _progress("PROJECTION", occurrences, truth_count, projection_started)
                last_progress_time = now
            if len(y_buf) >= shard_size:
                flush()

    flush()
    _progress("PROJECTION", occurrences, truth_count, projection_started)
    print(
        f"PROJECTION DONE | {occurrences:,}/{truth_count:,} | "
        f"ELAPSED {_fmt_duration(time.monotonic() - projection_started)} | {_resource_text()}",
        flush=True,
    )

    per_dimension = []
    zero_coverage = []
    partial_coverage = []
    for d in dimensions:
        k = int(d["index"])
        eligible = np.asarray(
            [s not in ("X", "K") for s in d["status_by_ascii"]],
            dtype=bool,
        )
        expected_chars = int(eligible.sum())
        covered_chars = int(np.count_nonzero(char_valid[:, k] > 0))
        item = {
            "index": k,
            "id": d["id"],
            "interaction": d["interaction"],
            "defer_scope": d["defer_scope"],
            "valid_occurrence_measurements": int(valid_counts[k]),
            "eligible_characters": expected_chars,
            "covered_characters": covered_chars,
            "character_coverage_fraction": (
                1.0 if expected_chars == 0 else covered_chars / expected_chars
            ),
            "minimum": None if not math.isfinite(min_value[k]) else float(min_value[k]),
            "maximum": None if not math.isfinite(max_value[k]) else float(max_value[k]),
        }
        per_dimension.append(item)
        if expected_chars > 0 and valid_counts[k] == 0:
            zero_coverage.append({"index": k, "id": d["id"]})
        elif expected_chars > 0 and covered_chars < expected_chars:
            partial_coverage.append(
                {
                    "index": k,
                    "id": d["id"],
                    "eligible_characters": expected_chars,
                    "covered_characters": covered_chars,
                }
            )

    report = {
        "format": "ascii95-scalar-evidence-projection-v1",
        "records_file": str(records.resolve()),
        "records_sha256": records_sha,
        "registry_file": str(registry_path.resolve()),
        "registry_sha256": registry_sha,
        "lexical_snapshot_file": (
            None if lexical_snapshot is None else str(lexical_snapshot.resolve())
        ),
        "lexical_snapshot_sha256": (
            None if lexical_snapshot is None else sha256_file(lexical_snapshot)
        ),
        "record_count": record_count,
        "ground_truth_occurrences": truth_count,
        "projected_occurrences": occurrences,
        "dimension_count": D,
        "checkpoint_file": str(_checkpoint_path(out_dir).resolve()),
        "checkpoint_resume_occurrence": resume_occurrence,
        "shards": shards,
        "strict_full_437_ready": len(zero_coverage) == 0 and len(partial_coverage) == 0,
        "zero_coverage_dimensions": zero_coverage,
        "partial_character_coverage_dimensions": partial_coverage,
        "per_dimension": per_dimension,
        "declared_projection_methods": {
            "registry_1_0": "literal fixed boolean evidence",
            "numeric_value": "literal decimal digit value",
            "orthographic": "deterministic measurement from literal occurrence/span",
            "corpus_unigram_and_position": "frequency over exact frozen records snapshot",
            "next_previous_probability": "observed adjacent pair conditional probability over exact snapshot",
            "next_entropy": "Shannon entropy of next-character distribution",
            "conditional_entropy": "reverse previous-character conditional entropy for current character",
            "evidence_counts": "counts/coverage over exact frozen snapshot",
            "ud_grammar_participation": "binary grammatical participation from aligned real UD dependency annotations",
            "lexical_semantic_participation": "binary lexeme/sense/relation participation from exact lexical snapshot rows",
            "python_programming_roles": "binary programming-role participation from Python tokenize annotations on real source lines",
        },
        "unresolved_policy": "NaN, never zero",
        "note": (
            "This projector resolves only scalar laws warranted by the attached "
            "real occurrence annotations and lexical snapshot. Categorical identities "
            "without a declared scalar encoding, pronunciation/phonetic features "
            "without grapheme-phone alignment, acoustic realization, glyph/rendering, "
            "and mathematical roles remain unresolved as NaN."
        ),
    }

    _atomic_json(out_dir / "binding_coverage.json", report)
    _atomic_json(
        out_dir / "projection_complete.json",
        {
            "format": "ascii95-projection-complete-v1",
            "records_sha256": records_sha,
            "registry_sha256": registry_sha,
            "lexical_snapshot_sha256": lexical_sha,
            "projected_occurrences": occurrences,
            "shard_count": len(shards),
            "coverage_report_sha256": sha256_file(out_dir / "binding_coverage.json"),
        },
    )

    if not report["strict_full_437_ready"] and not allow_incomplete:
        raise RuntimeError(
            "strict full-437 projection is incomplete; see binding_coverage.json"
        )
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument(
        "--registry",
        type=Path,
        default=Path("data/registry_contract.json"),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--lexical-snapshot",
        type=Path,
        default=None,
        help="assembler lexical_snapshot.jsonl; defaults to sibling of --records when present",
    )
    p.add_argument("--shard-size", type=int, default=100000)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be > 0")
    lexical_snapshot = args.lexical_snapshot
    if lexical_snapshot is None:
        candidate = args.records.parent / "lexical_snapshot.jsonl"
        if candidate.exists():
            lexical_snapshot = candidate
    try:
        report = build(
            args.records,
            args.registry,
            args.out,
            args.shard_size,
            args.allow_incomplete,
            lexical_snapshot,
        )
    except KeyboardInterrupt:
        print(f"PROJECTION BLOCKED | INTERRUPTED BY USER | {_resource_text()}", flush=True)
        return 130
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"PROJECTION FAILED | {exc} | {_resource_text()}", flush=True)
        return 1
    print(
        json.dumps(
            {
                "projected_occurrences": report["projected_occurrences"],
                "shard_count": len(report["shards"]),
                "strict_full_437_ready": report["strict_full_437_ready"],
                "zero_coverage_dimension_count": len(report["zero_coverage_dimensions"]),
                "partial_character_coverage_dimension_count": len(
                    report["partial_character_coverage_dimensions"]
                ),
                "coverage_report": str(args.out / "binding_coverage.json"),
            },
            indent=2,
        )
    )
    status = "DONE" if report["strict_full_437_ready"] else "BLOCKED"
    print(
        f"{status} | zero={len(report['zero_coverage_dimensions'])} | "
        f"partial={len(report['partial_character_coverage_dimensions'])} | "
        f"{_resource_text()}",
        flush=True,
    )
    return 0 if report["strict_full_437_ready"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
