# SQL-IR compiled model

## Run

Compile the user's read-only SQLite database once:

```sh
python -m sqlir.compile --db PATH --workbook sqlir/conical437_real_workbook.json --out MODEL_DIR
python -m sqlir.verify --model MODEL_DIR --db PATH --text Earth
python -m sqlir.serve --model MODEL_DIR
```

The server accepts JSON-lines requests on stdin/stdout by default. Use
`python -m sqlir.serve --model MODEL_DIR --http PORT` for a local HTTP endpoint
bound to `127.0.0.1`. For a fast channel-only compile, add `--only-channels`.
Lookup requests accept `offset` and `limit` for pagination; use the `count` operation to get a result count without decoding rows.
`sqlir_exe.py` dispatches `serve`, `verify`, and `compile` for the optional
PyInstaller executable; the artifact directory remains external to the exe.

## Explore the lexicon

`explore.py` runs a continuous, resumable discovery pass over the source
SQLite (opened `mode=ro` with `query_only`), including the `sx_` syllable
layer. State lives in a separate ledger SQLite; rerun the same command to
resume, and `--watch` re-explores when the source file changes.

```sh
python -m sqlir.explore run --db PATH --ledger LEDGER.sqlite --report report.md
python -m sqlir.explore run --db PATH --ledger LEDGER.sqlite --watch --report report.md
python -m sqlir.explore report --ledger LEDGER.sqlite --out report.md
```

Probes come from the live schema (row counts, column profiles, top values,
foreign-key orphans) plus syllable invariants (span contiguity/text/coverage,
source-record reconstruction, character rows, one nucleus per phonetic
syllable), Webster-vs-CMU syllable count and primary-stress comparisons, BEEP
onset/nucleus/coda structure, phone inventories, 437-workbook codes weighted by
letter tokens in syllables, and a frequency-driven walk of the syllable graph.
Every probe stores the SHA-256 of its full result plus a bounded sample; the
report records the source and schema SHA-256.

## Files

- `catalog.py` introspects SQLite schemas and creates the query catalog.
- `compile.py` creates the aligned `blob.bin` and its `manifest.json`.
- `runtime.py` memory-maps the artifact and implements lookup and forward calls.
- `verify.py` compares compiled lookups and forward results against live inputs.
- `serve.py` provides JSON-lines and HTTP request interfaces.
- `conical_sql_ir_transformer.py` and `conical437_real_workbook.json` are the
  user's source inputs, copied unchanged.
- `explore.py` is the read-only lexicon/syllable explorer and report writer.
- `tests/test_sqlir.py` and `tests/test_explore.py` contain only software test
  fixtures.

## Honest boundaries

- The artifact is a frozen, read-only precompiled form of the database, not new
  knowledge.
- Index segments duplicate rowids, so the artifact may be larger than the
  database.
- The conical forward pass's semantic limits are those stated in the user's
  README: character-histogram attribution is not proof of linguistic semantics,
  this is not an equivalent trained language model, and nearest-character
  output is a diagnostic rather than a validated decoder.
- Explorer findings are observations over the source as stored: workbook
  status codes (`X`, `E`, `R`, `M`, `U`, ...) are counted, never resolved; BEEP
  candidate examples are word membership, not letter-aligned segmentation; the
  stress comparison uses the first CMU pronunciation per lexeme.
- The SQLite test fixtures are test-only and must never be compiled into a model
  artifact for use as data.
