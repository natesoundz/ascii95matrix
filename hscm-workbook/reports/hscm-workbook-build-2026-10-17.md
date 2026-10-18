# HSCM × Workbook — Architecture Read + First Build (2026-10-17)

## What the original HSCM actually is (from the code)

- `matrix.py` declares the topology, nothing computed: **L0 = 65 NSM
  semantic primitives** (THING, MOVE, THINK, NOT...) + L1 = 45 sub-POS +
  L2 = 28 phonetic operators + L3 = 14 geometric keys + L4 = 10
  trajectories = the "162 KPI" layer. Sparse adjacency
  VALID_L0_L1_EDGES: "everything not listed here is zero."
- `engine.py`: a word's Coordinate is the tuple (L0, L1, L2 sequence,
  L3, L4 set) — the tuple IS the address; TraversalEngine.query does
  partial-tuple constraint intersection. PhoneticParser maps letters →
  L2 operators via CHAR_MAP/DIGRAPH_MAP.
- `manifold_engine.cpp`: 5-layer exhaustion builder; PHONEME_IDS maps
  the 39 ARPABET phones to L2 operator ids (vowels folded to 4 classes).
- `config.py`: DictPointer — file pointers + lazy word→byte-offset
  index; "never loads full files, sets pointers only."
- Honest stubs the author left: `_assign_primitive_subpos()`
  (orchestrator.py) is a word-length heuristic awaiting a real POS/sense
  source; L3 etymology expansion accepts all 14 keys.

## Two findings Nate should know

1. **The sparse version is not in Drive.** Both zip READMEs say the
   computational line is "derived from the existing HSCM Pure
   sparse-manifold engine / hscm_sparse.cpp" — that file is in neither
   the folder nor any zip. The zips are a different lineage (packed
   uint32 coordinates + mmap + coordinates carrying bytecode programs;
   verified working demo in FINAL_TEST_OUTPUT.json). The sparse engine
   itself must come from Nate's machine.
2. **Latent defect:** C++ `TrajSet` is uint8_t but trajectory masks run
   0..1023 (10 bits) — axis-4 poles are silently truncated in the C++
   manifold; the Python engine disagrees with it on L4. Also
   `hscm_full.py` imports three names that don't exist in the shipped
   matrix.py (it was written against a newer matrix.py).

## The first build (originals untouched)

Working copy: hidden_files/hscm_workbook/build/

- **workbook_l0.py** — the workbook as L0. Loads all 437 dimensions
  (id, name, section, interaction, eligibility, defer scope, per-char
  status) from the registry export; verifies against the manifest.
  Constraint API: status(char, dim); bindable() = hard gate on X;
  requires_evidence() for E/M/R/U/V/S cells. Sparse pointer view:
  pointers(char) = only the non-X dimensions (397 for 'a', 247 for '~').
  Status distribution of the loaded layer matches the workbook audit
  exactly (E 17,983 / X 10,110 / U 6,586 / M 2,220 / R 2,092 / ...).
- **phonetic_bridge.py** — the bridge Nate specified. An authored
  phone→feature table decomposes all 39 ARPABET phones into workbook
  phonetic dimension ids (standard articulatory phonetics; every id
  asserted to exist in the registry). A LexiconPointer indexes the
  lexicon export by lexeme_id; spellings join to pronunciations;
  alignment is positional over the strict 1:1 subset (spelling length
  == phone count, where alignment is unambiguous).

## Bridge results (real data, full run)

- 135,166 pronunciation rows processed; 34,223 lexemes in the aligned
  subset; 91,611 skipped (length mismatch — alignment would be a
  guess); 9,311 lemma skips; 21 dirty-phone rows skipped.
- **240 candidate (letter, dimension) binding cells across all 26
  letters; 158 of them strong** (≥90% of the letter's aligned phone
  tokens carry the feature). Examples: 'p' → bilabial, stop, plosive,
  voiceless, labial, anterior, consonantal (all strong). 'c' →
  voiceless and consonantal strong, but place only partial — correct:
  'c' realizes both K and S, which disagree on place, and the tiering
  refuses to call that a clean bind.
- **Gate violations: 0.** Every evidence-derived candidate respects
  the workbook's X gates — the registry and 135k pronunciations agree
  everywhere the bridge could test them.

Output with full provenance (support counts, shares, top phones,
registry status per cell): build/bridge_output.json.

## Next steps this sets up

1. Replace the `_assign_primitive_subpos` heuristic with the sqlite
   POS/sense/classification tables (the upstream data it was waiting
   for) — L1 becomes evidence-derived.
2. Relax the 1:1 alignment with a real grapheme↔phone aligner to widen
   coverage beyond 34k lexemes, keeping the same tier + gate law.
3. Port the L0 constraint layer into the sparse engine once
   hscm_sparse.cpp is available; fix the TrajSet width defect in the
   C++ line.
