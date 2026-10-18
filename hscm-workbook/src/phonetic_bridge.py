"""
Phonetic bridge: bind the workbook's 103 phonetic dimensions from the
lexicon database's pronunciation evidence, through HSCM-style pointers.

Chain (every link is a pointer, nothing is copied into the manifold):
  form/lexeme (spelling) --lexeme_id--> pronunciation (ARPABET phones)
  phone --PHONE_FEATURES--> workbook dimension ids (registry pointers)
  letter <--1:1 alignment-- phone        (only where spelling length ==
                                          phone count: positional alignment
                                          is then unambiguous)

PHONE_FEATURES is an authored DEFINITION table (standard articulatory
phonetics), not a measurement: each of the 39 ARPABET phones is assigned
the workbook dimension ids whose features it positively has. Absence of
a feature is never asserted here — the registry's own cells govern that.

Binding candidates are reported with provenance (support counts) and in
tiers, because letters are genuinely ambiguous ('c' -> K and S): a
feature binds STRONGLY for a letter only if >=90% of its aligned phone
tokens carry it; 30-90% is PARTIAL (real evidence, not a clean bind);
below that it is not a candidate. Gate law: no candidate may exist
where the registry says X (structurally impossible) — violations are
counted and reported, never silently dropped.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from workbook_l0 import WorkbookL0

DATA = Path(__file__).parent / "data"
P = "phonetic."  # namespace prefix

def _ids(*suffixes):
    return [P + s for s in suffixes]

# --- phone -> workbook phonetic dimension ids (positive features only) ---
_G = "phonetic_articulatory_general."
_V = "vowel_tongue_body_position."
_PL = "consonant_place_of_articulation."
_M = "consonant_manner_of_articulation."
_L = "laryngeal_phonation."

CONSONANT_COMMON = _ids("airstream.pulmonic")
VOWEL_COMMON = _ids("airstream.pulmonic", _G + "syllabic", _G + "sonorant")

PHONE_FEATURES = {
    # stops
    "P":  _ids(_G+"consonantal", _G+"anterior", _PL+"bilabial", _PL+"labial", _PL+"anterior", _M+"stop", _M+"plosive", _L+"voiceless"),
    "B":  _ids(_G+"consonantal", _G+"anterior", _G+"voice", _PL+"bilabial", _PL+"labial", _PL+"anterior", _M+"stop", _M+"plosive", _L+"voiced"),
    "T":  _ids(_G+"consonantal", _G+"anterior", _G+"coronal", _PL+"alveolar", _PL+"coronal", _PL+"anterior", _M+"stop", _M+"plosive", _L+"voiceless"),
    "D":  _ids(_G+"consonantal", _G+"anterior", _G+"coronal", _G+"voice", _PL+"alveolar", _PL+"coronal", _PL+"anterior", _M+"stop", _M+"plosive", _L+"voiced"),
    "K":  _ids(_G+"consonantal", _G+"back", _PL+"velar", _PL+"dorsal", _M+"stop", _M+"plosive", _L+"voiceless"),
    "G":  _ids(_G+"consonantal", _G+"back", _G+"voice", _PL+"velar", _PL+"dorsal", _M+"stop", _M+"plosive", _L+"voiced"),
    # fricatives
    "F":  _ids(_G+"consonantal", _G+"continuant", _PL+"labiodental", _PL+"labial", _M+"fricative", _L+"voiceless"),
    "V":  _ids(_G+"consonantal", _G+"continuant", _G+"voice", _PL+"labiodental", _PL+"labial", _M+"fricative", _L+"voiced"),
    "TH": _ids(_G+"consonantal", _G+"continuant", _G+"anterior", _G+"coronal", _PL+"dental", _PL+"coronal", _PL+"anterior", _M+"fricative", _L+"voiceless"),
    "DH": _ids(_G+"consonantal", _G+"continuant", _G+"anterior", _G+"coronal", _G+"voice", _PL+"dental", _PL+"coronal", _PL+"anterior", _M+"fricative", _L+"voiced"),
    "S":  _ids(_G+"consonantal", _G+"continuant", _G+"anterior", _G+"coronal", _G+"strident", _PL+"alveolar", _PL+"coronal", _PL+"anterior", _M+"fricative", _M+"sibilant", _L+"voiceless"),
    "Z":  _ids(_G+"consonantal", _G+"continuant", _G+"anterior", _G+"coronal", _G+"strident", _G+"voice", _PL+"alveolar", _PL+"coronal", _PL+"anterior", _M+"fricative", _M+"sibilant", _L+"voiced"),
    "SH": _ids(_G+"consonantal", _G+"continuant", _G+"coronal", _G+"distributed", _G+"strident", _PL+"postalveolar", _PL+"coronal", _PL+"distributed", _M+"fricative", _M+"sibilant", _L+"voiceless"),
    "ZH": _ids(_G+"consonantal", _G+"continuant", _G+"coronal", _G+"distributed", _G+"strident", _G+"voice", _PL+"postalveolar", _PL+"coronal", _PL+"distributed", _M+"fricative", _M+"sibilant", _L+"voiced"),
    "HH": _ids(_G+"continuant", _G+"spread_glottis", _PL+"glottal", _PL+"laryngeal", _M+"fricative", _L+"voiceless", _L+"spread_glottis"),
    # affricates
    "CH": _ids(_G+"consonantal", _G+"coronal", _G+"distributed", _G+"strident", _G+"delayed_release", _PL+"postalveolar", _PL+"coronal", _PL+"distributed", _M+"affricate", _L+"voiceless"),
    "JH": _ids(_G+"consonantal", _G+"coronal", _G+"distributed", _G+"strident", _G+"delayed_release", _G+"voice", _PL+"postalveolar", _PL+"coronal", _PL+"distributed", _M+"affricate", _L+"voiced"),
    # nasals
    "M":  _ids(_G+"consonantal", _G+"sonorant", _G+"nasal", _G+"voice", _PL+"bilabial", _PL+"labial", _M+"nasal", _L+"voiced"),
    "N":  _ids(_G+"consonantal", _G+"sonorant", _G+"nasal", _G+"anterior", _G+"coronal", _G+"voice", _PL+"alveolar", _PL+"coronal", _PL+"anterior", _M+"nasal", _L+"voiced"),
    "NG": _ids(_G+"consonantal", _G+"sonorant", _G+"nasal", _G+"back", _G+"voice", _PL+"velar", _PL+"dorsal", _M+"nasal", _L+"voiced"),
    # liquids / glides
    "L":  _ids(_G+"consonantal", _G+"sonorant", _G+"lateral", _G+"anterior", _G+"coronal", _G+"voice", _PL+"alveolar", _PL+"coronal", _PL+"anterior", _M+"approximant", _M+"lateral_approximant", _L+"voiced"),
    "R":  _ids(_G+"consonantal", _G+"sonorant", _G+"coronal", _G+"voice", _PL+"alveolar", _PL+"coronal", _M+"approximant", _L+"voiced"),
    "W":  _ids(_G+"sonorant", _G+"round", _G+"voice", _PL+"labial_velar", _PL+"labial", _PL+"dorsal", _M+"approximant", _L+"voiced"),
    "Y":  _ids(_G+"sonorant", _G+"high", _G+"voice", _PL+"palatal", _M+"approximant", _L+"voiced"),
    # vowels
    "IY": _ids(_G+"high", _G+"tense", _V+"close", _V+"front", _V+"unrounded"),
    "IH": _ids(_G+"high", _V+"near_close", _V+"near_front", _V+"unrounded"),
    "EH": _ids(_V+"open_mid", _V+"front", _V+"unrounded"),
    "AE": _ids(_V+"near_open", _V+"front", _V+"unrounded"),
    "AA": _ids(_G+"low", _G+"back", _V+"open", _V+"back", _V+"unrounded"),
    "AO": _ids(_G+"back", _G+"round", _V+"open_mid", _V+"back", _V+"rounded"),
    "OW": _ids(_G+"back", _G+"round", _G+"tense", _V+"close_mid", _V+"back", _V+"rounded"),
    "UH": _ids(_V+"near_close", _V+"near_back", _V+"rounded"),
    "UW": _ids(_G+"high", _G+"back", _G+"round", _G+"tense", _V+"close", _V+"back", _V+"rounded"),
    "ER": _ids(_V+"mid", _V+"central"),
    "AH": _ids(_V+"mid", _V+"central", _V+"unrounded"),
    "AY": _ids(_G+"low", _V+"open"),
    "AW": _ids(_G+"low", _V+"open"),
    "EY": _ids(_G+"tense", _V+"close_mid", _V+"front", _V+"unrounded"),
    "OY": _ids(_G+"round", _V+"open_mid", _V+"back", _V+"rounded"),
}
for ph, feats in PHONE_FEATURES.items():
    common = VOWEL_COMMON if ph in {"IY","IH","EH","AE","AA","AO","OW","UH","UW","ER","AH","AY","AW","EY","OY"} else CONSONANT_COMMON
    PHONE_FEATURES[ph] = feats + [f for f in common if f not in feats]

PHONES = set(PHONE_FEATURES)


class LexiconPointer:
    """Byte-offset pointer index over the lexicon JSONL export —
    the DictPointer pattern applied to the sqlite export tables."""
    def __init__(self, data_dir=DATA):
        self.data_dir = Path(data_dir)
        self._lemma = {}       # lexeme_id -> normalized lemma (offset-indexed lazily is overkill at 590k; ids are the pointers)
        with open(self.data_dir / "lexeme.jsonl") as f:
            for line in f:
                o = json.loads(line)
                self._lemma[o["lexeme_id"]] = o["normalized"]

    def lemma(self, lexeme_id):
        return self._lemma.get(lexeme_id)

    def pronunciations(self):
        with open(self.data_dir / "pronunciation.jsonl") as f:
            for line in f:
                yield json.loads(line)


def run(min_support=20):
    l0 = WorkbookL0()
    # every feature reference must be a real registry dimension
    bad_refs = [f for feats in PHONE_FEATURES.values() for f in feats
                if f not in l0.attributes]
    assert not bad_refs, f"feature table references unknown dims: {bad_refs[:5]}"

    lex = LexiconPointer()
    pair_counts = defaultdict(Counter)   # letter -> Counter(phone)
    stats = Counter()
    for row in lex.pronunciations():
        stats["pronunciation_rows"] += 1
        lemma = lex.lemma(row["lexeme_id"])
        if not lemma or not lemma.isalpha() or not lemma.islower():
            stats["skipped_lemma"] += 1
            continue
        raw = (row.get("phones") or "").split()
        phones = [t.rstrip("012") for t in raw]
        if any(t not in PHONES for t in phones):
            stats["skipped_dirty_phones"] += 1
            continue
        if len(lemma) != len(phones):
            stats["skipped_not_1to1"] += 1
            continue
        stats["aligned_lexemes"] += 1
        for ch, ph in zip(lemma, phones):
            pair_counts[ch][ph] += 1

    bindings = {}      # char -> {dim_id: {...}}
    violations = []
    for ch, phones in sorted(pair_counts.items()):
        total = sum(phones.values())
        feat_support = Counter()
        for ph, n in phones.items():
            for f in PHONE_FEATURES[ph]:
                feat_support[f] += n
        entry = {}
        for dim_id, sup in feat_support.items():
            share = sup / total
            tier = "strong" if share >= 0.9 else "partial" if share >= 0.3 else None
            if tier is None or sup < min_support:
                continue
            rec = {"support": sup, "total": total, "share": round(share, 3),
                   "tier": tier, "registry_status": l0.status(ch, dim_id),
                   "top_phones": [p for p, _ in phones.most_common(3)]}
            if not l0.bindable(ch, dim_id):
                violations.append({"char": ch, "dim": dim_id, **rec})
            entry[dim_id] = rec
        if entry:
            bindings[ch] = entry

    out = {"stats": dict(stats),
           "letters_with_candidates": len(bindings),
           "candidate_cells": sum(len(v) for v in bindings.values()),
           "strong_cells": sum(1 for v in bindings.values() for r in v.values() if r["tier"] == "strong"),
           "gate_violations": violations,
           "bindings": bindings}
    Path(__file__).with_name("bridge_output.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: out[k] for k in
                      ("stats", "letters_with_candidates", "candidate_cells", "strong_cells")}, indent=1))
    print("gate violations:", len(out["gate_violations"]))
    for ch in ("p", "a", "c"):
        b = out["bindings"].get(ch, {})
        strong = [d.split(".")[-1] for d, r in b.items() if r["tier"] == "strong"]
        print(f"  '{ch}': {len(b)} candidates, strong: {sorted(strong)[:10]}")
