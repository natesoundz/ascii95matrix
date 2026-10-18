"""Aggregate the sqlite classification tables into a per-lemma index.

Sources (the export tables Nate designated):
  sense.jsonl                    lexeme -> POS counts, WordNet lexical classes
  evidence (filtered extract)    lexeme -> DOMINANT_POS_NORM, semantic classes,
                                 COUNTABILITY, CONCRETENESS, adj/adv class
  occurrence.jsonl               token-level UD POS per lemma (real usage)
  occurrence_classification.jsonl warranted PROPER_NOUN_USE counts per lemma

Output: classifier_index.json keyed by normalized lemma. Counts are kept,
not collapsed — the classifier's rules read the distributions, and every
assignment can cite its evidence.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data"


def main():
    lemma_of = {}
    with open(DATA / "lexeme.jsonl") as f:
        for line in f:
            o = json.loads(line)
            lemma_of[o["lexeme_id"]] = o["normalized"]

    idx = defaultdict(lambda: {
        "sense_pos": Counter(), "lex_classes": Counter(), "pos_ev": Counter(),
        "dominant_pos": None, "noun_sem": Counter(), "verb_sem": Counter(),
        "countability": Counter(), "concreteness": [],
        "adj_class": Counter(), "adv_class": Counter(),
        "ud_pos": Counter(), "propn": 0, "noun_tokens": 0, "proper_warrants": 0})

    with open(DATA / "sense.jsonl") as f:
        for line in f:
            o = json.loads(line)
            w = lemma_of.get(o["lexeme_id"])
            if not w:
                continue
            e = idx[w]
            e["sense_pos"][o["pos"]] += 1
            if o.get("lexical_class"):
                e["lex_classes"][o["lexical_class"]] += 1

    ev = json.load(open(DATA / "evidence_filtered.json"))
    for lex_id, cats in ev.items():
        w = lemma_of.get(int(lex_id))
        if not w:
            continue
        e = idx[w]
        for cat, rows in cats.items():
            for r in rows:
                sub = r["sub"]
                if cat == "DOMINANT_POS_NORM":
                    if sub not in ("#N/A", "0", "Unclassified"):
                        e["dominant_pos"] = sub
                elif cat == "NOUN_SEMANTIC_CLASS":
                    e["noun_sem"][sub] += 1
                elif cat == "VERB_SEMANTIC_CLASS":
                    e["verb_sem"][sub] += 1
                elif cat == "COUNTABILITY":
                    e["countability"][sub] += 1
                elif cat == "CONCRETENESS" and sub == "MEAN" and r["num"] is not None:
                    e["concreteness"].append(r["num"])
                elif cat == "ADJECTIVE_CLASS":
                    e["adj_class"][sub] += 1
                elif cat == "ADVERB_CLASS":
                    e["adv_class"][sub] += 1

    pos_ev = json.load(open(DATA / "evidence_pos.json"))
    for lex_id, counts in pos_ev.items():
        w = lemma_of.get(int(lex_id))
        if w:
            idx[w]["pos_ev"] = Counter(counts)

    # occurrence-level UD POS + proper-noun warrants
    occ = {}
    with open(DATA / "occurrence.jsonl") as f:
        for line in f:
            o = json.loads(line)
            w = o["normalized"]
            occ[o["occurrence_id"]] = w
            e = idx[w]
            e["ud_pos"][o["pos"]] += 1
            if o["pos"] == "PROPN":
                e["propn"] += 1
            if o["pos"] in ("NOUN", "PROPN"):
                e["noun_tokens"] += 1
    with open(DATA / "occurrence_classification.jsonl") as f:
        for line in f:
            o = json.loads(line)
            if o.get("warranted") == 1 and "PROPER" in (o.get("subcategory") or ""):
                w = occ.get(o["occurrence_id"])
                if w:
                    idx[w]["proper_warrants"] += 1

    # Form inheritance: a word with no classification of its own that
    # exists as a FORM (inflection/spelling) of a classified lexeme
    # inherits that lexeme's evidence, marked as inherited. This is how
    # the CMU pronunciation stratum (lexemes with pronunciations but no
    # senses/evidence) connects to the classified Webster stratum.
    form_owners = {}
    with open(DATA / "form.jsonl") as f:
        for line in f:
            o = json.loads(line)
            owner = lemma_of.get(o["lexeme_id"])
            if owner:
                form_owners.setdefault(o["normalized"], set()).add(owner)
    inherited = 0
    for form_str, owners in form_owners.items():
        if form_str in idx:
            continue
        classified = [o for o in owners if o in idx]
        if not classified:
            continue
        best = max(classified, key=lambda o: sum(idx[o]["sense_pos"].values())
                   + sum(idx[o]["pos_ev"].values()))
        entry = dict(idx[best])
        entry["inherited_from"] = best
        idx[form_str] = entry
        inherited += 1
    print("form-inherited entries:", inherited)

    out = {}
    for w, e in idx.items():
        out[w] = {
            "sense_pos": dict(e["sense_pos"]), "lex_classes": dict(e["lex_classes"]),
            "pos_ev": dict(e["pos_ev"]),
            "dominant_pos": e["dominant_pos"], "noun_sem": dict(e["noun_sem"]),
            "verb_sem": dict(e["verb_sem"]), "countability": dict(e["countability"]),
            "concreteness": (sum(e["concreteness"]) / len(e["concreteness"])) if e["concreteness"] else None,
            "adj_class": dict(e["adj_class"]), "adv_class": dict(e["adv_class"]),
            "ud_pos": dict(e["ud_pos"]), "propn": e["propn"],
            "noun_tokens": e["noun_tokens"], "proper_warrants": e["proper_warrants"],
            "inherited_from": e.get("inherited_from")}
    json.dump(out, open(DATA / "classifier_index.json", "w"))
    print("lemmas indexed:", len(out))
    have_dom = sum(1 for v in out.values() if v["dominant_pos"])
    have_sense = sum(1 for v in out.values() if v["sense_pos"])
    have_ud = sum(1 for v in out.values() if v["ud_pos"])
    print(f"with DOMINANT_POS_NORM: {have_dom} | with senses: {have_sense} | with UD occurrences: {have_ud}")


if __name__ == "__main__":
    main()
