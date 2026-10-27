"""TEST-ONLY fixture: validates the explorer software only; fixture rows are not lexicon data."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sqlir.explore import build_report, explore, main, open_source, parse_record


SCHEMA = """
CREATE TABLE lexeme (lexeme_id INTEGER PRIMARY KEY, lemma TEXT, normalized TEXT);
CREATE TABLE pronunciation (pronunciation_id INTEGER PRIMARY KEY,
  lexeme_id INTEGER REFERENCES lexeme(lexeme_id), phones TEXT, stress_pattern TEXT, syllable_count INTEGER);
CREATE TABLE sx_word_variant (variant_id INTEGER PRIMARY KEY, lexeme_id INTEGER REFERENCES lexeme(lexeme_id),
  spelling TEXT, source_record TEXT, source_key TEXT, alignment_status TEXT);
CREATE TABLE sx_syllable (syllable_id INTEGER PRIMARY KEY, kind TEXT, notation TEXT, source_key TEXT,
  analysis_status TEXT, text TEXT);
CREATE TABLE sx_word_syllable (variant_id INTEGER REFERENCES sx_word_variant(variant_id), ordinal INTEGER,
  start_char INTEGER, end_char INTEGER, syllable_id INTEGER REFERENCES sx_syllable(syllable_id),
  PRIMARY KEY(variant_id, ordinal));
CREATE TABLE sx_syllable_character (syllable_id INTEGER REFERENCES sx_syllable(syllable_id), ordinal INTEGER,
  codepoint INTEGER, PRIMARY KEY(syllable_id, ordinal));
CREATE TABLE sx_character (codepoint INTEGER PRIMARY KEY, character TEXT);
CREATE TABLE sx_attribute (attribute_id TEXT PRIMARY KEY, section TEXT, is_physical INTEGER);
CREATE TABLE sx_character_attribute (codepoint INTEGER, attribute_id TEXT, value_json TEXT,
  PRIMARY KEY(codepoint, attribute_id));
CREATE TABLE sx_phone (syllable_id INTEGER REFERENCES sx_syllable(syllable_id), ordinal INTEGER, symbol TEXT,
  role TEXT, PRIMARY KEY(syllable_id, ordinal));
CREATE TABLE sx_candidate_example (syllable_id INTEGER, example_word TEXT, lexeme_id INTEGER);
"""


def create_fixture(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO lexeme VALUES (?,?,?)", [(1, "ABACA", "abaca"), (2, "BIN", "bin")])
    con.executemany(
        "INSERT INTO pronunciation VALUES (?,?,?,?,?)",
        [(1, 1, "AE1 B AH0 K AH0", "100", 3), (2, 2, "B IH1 N", "1", 1)],
    )
    con.executemany(
        "INSERT INTO sx_word_variant VALUES (?,?,?,?,?,?)",
        [(1, 1, "ABACA", 'ABACA=Ab"a*ca', "webster_syllabification", "aligned"),
         (2, 2, "BIN", "BIN=Bin", "webster_syllabification", "aligned")],
    )
    syllables = {1: "AB", 2: "A", 3: "CA", 4: "BIN"}
    con.executemany(
        "INSERT INTO sx_syllable VALUES (?,?,?,?,?,?)",
        [(k, "orthographic", "ASCII95", "webster_syllabification", "source_reported", v)
         for k, v in syllables.items()]
        + [(5, "phonetic", "BEEP", "barker_beep", "source_reported", "b ih n")],
    )
    con.executemany(
        "INSERT INTO sx_word_syllable VALUES (?,?,?,?,?)",
        [(1, 0, 0, 2, 1), (1, 1, 2, 3, 2), (1, 2, 3, 5, 3), (2, 0, 0, 3, 4)],
    )
    con.executemany(
        "INSERT INTO sx_syllable_character VALUES (?,?,?)",
        [(k, i, ord(ch)) for k, v in syllables.items() for i, ch in enumerate(v)],
    )
    letters = sorted({ch for v in syllables.values() for ch in v})
    con.executemany("INSERT INTO sx_character VALUES (?,?)", [(ord(ch), ch) for ch in letters])
    con.executemany("INSERT INTO sx_attribute VALUES (?,?,?)", [("A001", "01 Character Identity", 0)])
    con.executemany(
        "INSERT INTO sx_character_attribute VALUES (?,?,?)", [(ord(ch), "A001", '"E"') for ch in letters]
    )
    con.executemany(
        "INSERT INTO sx_phone VALUES (?,?,?,?)", [(5, 0, "b", "onset"), (5, 1, "ih", "nucleus"), (5, 2, "n", "coda")]
    )
    con.execute("INSERT INTO sx_candidate_example VALUES (5, 'bin', 2)")
    con.commit()
    con.close()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quiet(*args, **kwargs):
    lines: list[str] = []
    state = explore(*args, log=lines.append, **kwargs)
    return state, lines


class ExploreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = root / "fixture.sqlite"
        self.ledger = root / "ledger.sqlite"
        create_fixture(self.db)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def statuses(self) -> dict[str, str]:
        con = sqlite3.connect(self.ledger)
        try:
            return dict(con.execute("SELECT probe_id, status FROM probe"))
        finally:
            con.close()

    def test_full_run_passes_invariants_and_leaves_source_untouched(self) -> None:
        before = sha(self.db)
        state, _ = quiet(self.db, self.ledger)
        self.assertEqual(sha(self.db), before)
        statuses = self.statuses()
        self.assertNotIn("pending", statuses.values())
        self.assertEqual({s for s in statuses.values()}, {"done"})
        report = build_report(self.ledger, state)
        for name in ("span_gaps", "span_text", "span_cover", "record_reconstructs", "record_headword",
                     "char_rows", "char_count", "phonetic_nucleus", "phone_role_order"):
            self.assertIn(f"| syll.inv:{name} | PASS |", report)
        self.assertNotIn("| FAIL |", report)
        self.assertIn("Webster and CMU syllable counts agree for 2/2 variants", report)

    def test_rerun_resumes_and_source_change_creates_new_state(self) -> None:
        state1, _ = quiet(self.db, self.ledger, max_probes=3)
        done = [k for k, v in self.statuses().items() if v == "done"]
        self.assertEqual(len(done), 3)
        state2, lines = quiet(self.db, self.ledger)
        self.assertEqual(state1, state2)
        self.assertIn("resumed", lines[0])
        self.assertIn("seeded 0 new probes", lines[0])
        con = sqlite3.connect(self.db)
        con.execute("UPDATE sx_word_syllable SET end_char = 4 WHERE variant_id = 2")
        con.commit()
        con.close()
        state3, _ = quiet(self.db, self.ledger)
        self.assertNotEqual(state3, state1)
        report = build_report(self.ledger, state3)
        self.assertIn("| syll.inv:span_cover | FAIL |", report)
        self.assertIn("| syll.inv:span_text | FAIL |", report)

    def test_source_connection_is_read_only(self) -> None:
        con = open_source(self.db)
        try:
            with self.assertRaises(sqlite3.Error):
                con.execute("DELETE FROM lexeme")
        finally:
            con.close()

    def test_cli_run_and_report(self) -> None:
        out = self.ledger.with_suffix(".md")
        with redirect_stdout(io.StringIO()):
            main(["run", "--db", str(self.db), "--ledger", str(self.ledger), "--report", str(out)])
        text = out.read_text(encoding="utf-8")
        self.assertIn("# Lexicon exploration report", text)
        self.assertIn(sha(self.db), text)

    def test_parse_record(self) -> None:
        self.assertEqual(parse_record('ABACA=Ab"a*ca'), [("Ab", "primary"), ("a", "break"), ("ca", "end")])
        self.assertEqual(
            parse_record("ABACINATION=A*bac`i*na\"tion"),
            [("A", "break"), ("bac", "secondary"), ("i", "break"), ("na", "primary"), ("tion", "end")],
        )
        self.assertEqual(parse_record("BIN=Bin*"), [("Bin", "break")])
        self.assertEqual(parse_record("ABLE=*a*ble"), [("a", "break"), ("ble", "end")])


if __name__ == "__main__":
    unittest.main()
