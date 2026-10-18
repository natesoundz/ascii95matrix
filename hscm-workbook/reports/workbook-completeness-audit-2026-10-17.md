# Canonical ASCII95 Workbook — Completeness Audit (2026-10-17)

Question (Nate): is the Canonical ASCII95 Attribute Matrix workbook
complete, and if not, exactly what is missing? In particular the claim he
has been given that not all 437 columns carry their labels (the name of
the feature/attribute/dimension).

Copies examined (cell-by-cell):
- Google Drive: Canonical_ASCII95_Attribute_Matrix_v1.xlsm
  (156,240 bytes, modified 2026-09-27)
- GitHub ascii95matrix repo: data/Canonical_ASCII95_Attribute_Matrix_v1.xlsx
  SHA-256 = 065cb5424ec0fd2d206de093e13ed2317f29a0d6a1eda837df1a9d46f81506ab
  — matches the canonical hash recorded in the repo's status document.
- The two copies are CONTENT-IDENTICAL: 44,677 nonempty cells in the
  Matrix sheet and 57 in the Legend sheet in both; all 192 differing
  cells are numeric formatting only ('38' vs '38.0'); zero substantive
  differences.

## Findings — the registry is complete

Matrix sheet layout: title row; 7 column-metadata rows (Section,
Interaction, Eligibility, Positive/Candidate, Defer/Scope, Display Name,
Canonical ID); header row; 95 character rows (rows for all printable
ASCII characters, labels present, first [space] last ~).

- Display Name (the label of each dimension): present for 437 / 437
  columns. Zero missing.
- Canonical ID: present for 437 / 437 columns. Zero missing.
- Interaction type: present for 437 / 437. Eligibility: 437 / 437.
- Workbook Display Names and Canonical IDs match
  data/registry_dimensions.tsv exactly — 437 / 437, zero mismatches.
- Data grid: 41,515 / 41,515 cells filled (437 x 95) — every dimension
  column carries a status code for every character. Zero empty data
  cells; zero entirely empty columns.
  Code distribution: E 17,983; X 10,110; U 6,586; M 2,220; R 2,092;
  0: 806; V: 780; K: 380; 1: 328; S: 230.
- Section headers: 28 group labels at group-start columns
  (01 Character Identity ... 22 Source Classes) — group headers by
  design, not per-column data.
- Defer/Scope metadata filled for 399 columns, matching the TSV's 399.
  Positive/Candidate metadata filled for 62 columns (applicable-only).
- Legend sheet complete: all 10 code meanings, matrix properties
  (95 rows, 437 columns, first attribute column F, last PZ), and the
  permanent invariants.

Verdict on the claim Nate was given: FALSE. No column is missing its
label or symbol-name. Every dimension is named, twice (Display Name and
Canonical ID), in both surviving copies.

## What is actually incomplete — evidence binding, by design

The workbook's data cells are STATUS codes, not measurements. The
largest class, E (17,983 cells), means "eligible / deferred — can exist
but requires context/source/defer-axis resolution." Per the workbook's
own invariants ("Eligibility is not activation"; "Missing/undefined is
not zero"), those cells are deliberately unbound, not missing.

Quantified in ascii95matrix/REAL_COMPILED_TRANSFORMER_STATUS.md:
212 / 437 dimensions have real bound evidence; 177 are geometrically
active; 225 dimensions remain unresolved — the largest blocks are
phonetic/articulatory, pronunciation, acoustics, and
lexical/morphology/grammar/semantic fields awaiting real annotation
sources. The named next source is webster_full_lexicon.sqlite.

So "completing the workbook" has exactly one meaningful sense left:
binding real evidence into the deferred (E), measurement (M), and
relation (R) dimensions and recompiling — the registry/label layer is
already finished. A 118 MB canonical_sqlite_evidence.zip (Drive,
2026-10-16) appears to be the evidence package for that step; it has
not yet been reconciled against the 225 unresolved dimensions.

Audit artifacts: hidden_files/workbook_audit/ (both copies, parser,
audit scripts, registry markdown).
