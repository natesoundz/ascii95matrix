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
`sqlir_exe.py` dispatches `serve`, `verify`, and `compile` for the optional
PyInstaller executable; the artifact directory remains external to the exe.

## Files

- `catalog.py` introspects SQLite schemas and creates the query catalog.
- `compile.py` creates the aligned `blob.bin` and its `manifest.json`.
- `runtime.py` memory-maps the artifact and implements lookup and forward calls.
- `verify.py` compares compiled lookups and forward results against live inputs.
- `serve.py` provides JSON-lines and HTTP request interfaces.
- `conical_sql_ir_transformer.py` and `conical437_real_workbook.json` are the
  user's source inputs, copied unchanged.
- `tests/test_sqlir.py` contains only a software test fixture.

## Honest boundaries

- The artifact is a frozen, read-only precompiled form of the database, not new
  knowledge.
- Index segments duplicate rowids, so the artifact may be larger than the
  database.
- The conical forward pass's semantic limits are those stated in the user's
  README: character-histogram attribution is not proof of linguistic semantics,
  this is not an equivalent trained language model, and nearest-character
  output is a diagnostic rather than a validated decoder.
- The SQLite test fixture is test-only and must never be compiled into a model
  artifact for use as data.
