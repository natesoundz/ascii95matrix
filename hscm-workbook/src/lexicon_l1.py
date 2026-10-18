"""
LexiconL1 — evidence-driven replacement for orchestrator.py's
_assign_primitive_subpos() heuristic stub.

Same contract:  assign(word, phonemes) -> (primitive, subpos) | None
(None when phonemes is falsy, exactly as the stub — the engine needs
the L2 sequence anyway.)

Every assignment is derived from the sqlite classification tables
(sense / evidence / occurrence_classification, aggregated in
classifier_index.json) by declared, deterministic rule ladders, and is
GUARANTEED valid: the pair must appear in matrix.py's
VALID_L0_L1_EDGES, which this module imports from the original file.

Where the evidence wants a subpos that the edge table cannot reach at
all (NOUN_PROPER, NOUN_MASS, NOUN_COLLECTIVE, NOUN_ROLE, ADJ_RELATIONAL,
ADJ_TEMPORAL, ADV_MANNER, ADV_FREQUENCY, PRONOUN_RELATIVE,
DETERMINER_DEF have NO incoming edges), the ladder degrades to the
nearest reachable pair and flags it `degraded` — the counts of those
flags are the precise size of the matrix gap, reported, not hidden.
Words with no evidence at all fall back to the original heuristic,
flagged `fallback`. Provenance for every call is available via
assign_with_provenance().
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "hscm_matrix", _HERE / ".." / "originals" / "matrix.py")
_matrix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_matrix)
EDGES = _matrix.VALID_L0_L1_EDGES
VALID_PAIRS = {(p, s) for p, subs in EDGES.items() for s in subs}
REACHABLE_SUBPOS = {s for _, s in VALID_PAIRS}

TEMPORAL_PREPS = {"before", "after", "during", "until", "since"}
RELATIVE_PRONOUNS = {"who", "whom", "whose", "which", "that"}
SPATIAL_ADJ = {"near", "far", "close", "distant", "upper", "lower", "inner", "outer"}
QUANT_ADJ = {"many", "much", "few", "little", "several", "enough", "more", "most", "less", "least"}
MODAL_ADJ = {"possible", "impossible", "necessary", "able", "unable", "likely", "unlikely", "certain"}
DEGREE_ADV = {"very", "too", "quite", "rather", "so", "more", "most", "less", "least", "almost", "nearly", "really", "just", "even"}
TEMPORAL_ADV = {"now", "then", "today", "tomorrow", "yesterday", "before", "after", "already", "still", "yet", "soon", "when"}
SPATIAL_ADV = {"here", "there", "where", "near", "far", "away", "back", "home", "abroad"}
MODAL_ADV = {"maybe", "perhaps", "probably", "possibly", "certainly", "not", "never", "always"}
FREQ_ADV = {"often", "sometimes", "usually", "rarely", "seldom", "frequently", "occasionally"}
EVAL_ADV = {"well", "badly", "fortunately", "unfortunately", "luckily", "hopefully"}
EXISTENCE_VERBS = {"be", "exist", "remain", "stay"}
AUDITORY_VERBS = {"hear", "listen", "sound"}

CONCRETE_NOUN_CLASSES = {"noun.artifact", "noun.object", "noun.substance", "noun.food",
                         "noun.plant", "noun.body", "noun.location", "noun.shape",
                         "noun.phenomenon"}
ABSTRACT_NOUN_CLASSES = {"noun.attribute", "noun.cognition", "noun.motive",
                         "noun.possession", "noun.quantity", "noun.relation", "noun.Tops"}


def _top(counter):
    return counter.most_common(1)[0][0] if counter else None


class LexiconL1:
    def __init__(self, index_path=None):
        self.index = json.load(open(index_path or (_HERE / "data" / "classifier_index.json")))
        self.stats = Counter()

    # ------------------------------------------------------------------
    def assign(self, word, phonemes=None):
        pair, _prov = self.assign_with_provenance(word, phonemes)
        return pair

    def assign_with_provenance(self, word, phonemes=None):
        if not phonemes:
            return None, {"rule": "no_phonemes"}
        w = (word or "").strip().lower()
        e = self.index.get(w)
        if e is None:
            self.stats["fallback_no_evidence"] += 1
            return self._fallback(w, phonemes), {"rule": "fallback_no_evidence"}
        pair, rule, degraded = self._classify(w, e)
        if pair is None:   # indexed but no usable classification signal
            self.stats["fallback_no_signal"] += 1
            return self._fallback(w, phonemes), {"rule": "fallback_no_signal"}
        assert pair in VALID_PAIRS, f"invalid pair emitted: {pair}"
        self.stats["degraded" if degraded else "evidence"] += 1
        if degraded:
            self.stats[f"degraded:{rule}"] += 1
        prov = {"rule": rule, "degraded": degraded}
        if e.get("inherited_from"):
            prov["inherited_from"] = e["inherited_from"]
            self.stats["via_form_inheritance"] += 1
        return pair, prov

    # ------------------------------------------------------------------
    def _classify(self, w, e):
        pos = self._dominant_pos(w, e)
        if pos == "NOUN":
            return self._noun(w, e)
        if pos == "VERB":
            return self._verb(w, e)
        if pos == "ADJ":
            return self._adj(w, e)
        if pos == "ADV":
            return self._adv(w, e)
        if pos == "PRON":
            if w in RELATIVE_PRONOUNS:
                return ("SOMEBODY", "PRONOUN_PERSONAL"), "pron_relative_unreachable", True
            return ("SOMEBODY", "PRONOUN_PERSONAL"), "pron_personal", False
        if pos == "ADP":
            if w in TEMPORAL_PREPS:
                prim = {"before": "BEFORE", "after": "AFTER"}.get(w, "BEFORE")
                return (prim, "PREPOSITION_TEMPORAL"), "prep_temporal", w not in ("before", "after")
            return ("HERE", "PREPOSITION_SPATIAL"), "prep_spatial_default", False
        if pos == "CONJ":
            ud = Counter(e["ud_pos"])
            if w == "if":
                return ("IF", "CONJUNCTION_SUB"), "conj_if", False
            if w in ("because", "since", "as"):
                return ("BECAUSE", "CONJUNCTION_SUB"), "conj_because", False
            if w in ("before", "after", "when", "while", "until"):
                return ("BEFORE" if w != "after" else "AFTER", "CONJUNCTION_SUB"), "conj_temporal", True
            if ud.get("CCONJ", 0) >= ud.get("SCONJ", 0):
                return ("LIKE", "CONJUNCTION_COORD"), "conj_coord_only_edge", True
            return ("BECAUSE", "CONJUNCTION_SUB"), "conj_sub_default", True
        if pos == "DET":
            return ("NOT", "PARTICLE"), "determiner_unreachable", True
        if pos == "PART":
            return ("NOT", "PARTICLE"), "particle", True
        if pos == "NUM":
            return ("MORE", "ADJ_QUANTITATIVE"), "number_as_quantitative", True
        return None, "no_pos", False

    def _dominant_pos(self, w, e):
        dom = e.get("dominant_pos")
        dom_map = {"Noun": "NOUN", "Verb": "VERB", "Adjective": "ADJ", "Adverb": "ADV",
                   "Preposition": "ADP", "Pronoun": "PRON", "Conjunction": "CONJ",
                   "Determiner": "DET", "Article": "DET", "Name": "NOUN",
                   "Number": "NUM", "Interjection": "PART", "To": "ADP",
                   "Letter": "NOUN", "Not": "PART", "Ex": "PART"}
        if dom in dom_map:
            return dom_map[dom]
        sp = Counter(e["sense_pos"])
        if sp:
            return {"NOUN": "NOUN", "VERB": "VERB", "ADJ": "ADJ", "ADV": "ADV"}.get(_top(sp))
        pe = Counter(e.get("pos_ev") or {})
        if pe:
            return {"NOUN": "NOUN", "PROPN": "NOUN", "VERB": "VERB", "AUX": "VERB",
                    "ADJ": "ADJ", "ADV": "ADV", "PRON": "PRON", "ADP": "ADP",
                    "CCONJ": "CONJ", "SCONJ": "CONJ", "DET": "DET", "PART": "PART",
                    "NUM": "NUM", "INTJ": "PART"}.get(_top(pe))
        ud = Counter(e["ud_pos"])
        if ud:
            return {"NOUN": "NOUN", "PROPN": "NOUN", "VERB": "VERB", "AUX": "VERB",
                    "ADJ": "ADJ", "ADV": "ADV", "PRON": "PRON", "ADP": "ADP",
                    "CCONJ": "CONJ", "SCONJ": "CONJ", "DET": "DET", "PART": "PART",
                    "NUM": "NUM"}.get(_top(ud))
        return None

    # ------------------------------------------------------------------
    def _noun(self, w, e):
        sem = Counter(e["noun_sem"]) or Counter(
            {k: v for k, v in e["lex_classes"].items() if k.startswith("noun.")})
        cls = _top(sem)
        count = Counter(e["countability"])
        conc = e.get("concreteness")
        propn_share = (e["propn"] / e["noun_tokens"]) if e["noun_tokens"] else 0.0
        animate = cls in ("noun.person", "noun.animal")

        warrant_share = (e["proper_warrants"] / e["noun_tokens"]) if e["noun_tokens"] else 0.0
        pe = Counter(e.get("pos_ev") or {})
        pe_total = sum(v for k, v in pe.items() if k in ("NOUN", "PROPN"))
        pe_propn_share = (pe.get("PROPN", 0) / pe_total) if pe_total else 0.0
        if e.get("dominant_pos") == "Name" or propn_share >= 0.5 or warrant_share >= 0.5 or pe_propn_share >= 0.5:
            if animate or cls == "noun.person":
                return ("SOMEBODY", "NOUN_ANIMATE"), "noun_proper_unreachable", True
            return ("THING", "NOUN_CONCRETE"), "noun_proper_unreachable", True
        if cls == "noun.group":
            return ("THING", "NOUN_CONCRETE"), "noun_collective_unreachable", True
        if count.get("regular_mass", 0) and not count.get("regular_count", 0):
            return ("THING", "NOUN_INANIMATE"), "noun_mass_unreachable", True
        if animate:
            return ("SOMEBODY", "NOUN_ANIMATE"), "noun_animate", False
        if cls == "noun.event":
            return ("HAPPEN", "NOUN_EVENT"), "noun_event", False
        if cls == "noun.act":
            return ("DO", "NOUN_EVENT"), "noun_act", False
        if cls == "noun.feeling":
            return ("FEEL", "NOUN_STATE"), "noun_feeling", False
        if cls == "noun.state":
            return ("KNOW", "NOUN_STATE"), "noun_state", False
        if cls == "noun.cognition":
            return ("KNOW", "NOUN_ABSTRACT"), "noun_cognition", False
        if cls == "noun.communication":
            return ("SAY", "NOUN_PROCESS"), "noun_communication", False
        if cls == "noun.process":
            return ("MOVE", "NOUN_PROCESS"), "noun_process_approx", True
        if cls == "noun.time":
            return ("THING", "NOUN_ABSTRACT"), "noun_time_abstract", True
        if cls in CONCRETE_NOUN_CLASSES or (conc is not None and conc >= 3.0):
            return ("THING", "NOUN_CONCRETE"), "noun_concrete", False
        if cls in ABSTRACT_NOUN_CLASSES or (conc is not None and conc < 2.5):
            return ("THING", "NOUN_ABSTRACT"), "noun_abstract", False
        if conc is not None:
            return ("THING", "NOUN_INANIMATE"), "noun_inanimate_by_concreteness", False
        return ("THING", "NOUN_CONCRETE"), "noun_default_concrete", True

    def _verb(self, w, e):
        sem = Counter(e["verb_sem"]) or Counter(
            {k: v for k, v in e["lex_classes"].items() if k.startswith("verb.")})
        cls = _top(sem)
        if w in EXISTENCE_VERBS:
            return ("BE", "VERB_EXISTENCE"), "verb_existence_lemma", False
        if w in AUDITORY_VERBS:
            return ("HEAR", "VERB_PERCEPTION"), "verb_auditory_lemma", False
        return {
            "verb.motion": ("MOVE", "VERB_MOTION"),
            "verb.communication": ("SAY", "VERB_SPEECH"),
            "verb.cognition": ("THINK", "VERB_MENTAL"),
            "verb.emotion": ("FEEL", "VERB_MENTAL"),
            "verb.perception": ("SEE", "VERB_PERCEPTION"),
            "verb.possession": ("HAVE", "VERB_POSSESSION"),
            "verb.change": ("BECOME", "VERB_CHANGE"),
            "verb.stative": ("BE", "VERB_STATE"),
            "verb.weather": ("HAPPEN", "VERB_PROCESS"),
            "verb.creation": ("DO", "VERB_ACTION"),
            "verb.contact": ("DO", "VERB_ACTION"),
            "verb.social": ("DO", "VERB_ACTION"),
            "verb.consumption": ("DO", "VERB_ACTION"),
            "verb.competition": ("DO", "VERB_ACTION"),
            "verb.body": ("DO", "VERB_ACTION"),
        }.get(cls, ("DO", "VERB_ACTION")), ("verb_" + (cls or "default")), cls is None

    def _adj(self, w, e):
        adj = Counter(e["adj_class"])
        if w in SPATIAL_ADJ:
            return ("NEAR" if w != "far" else "FAR", "ADJ_SPATIAL"), "adj_spatial_lemma", False
        if w in QUANT_ADJ:
            return ("MORE", "ADJ_QUANTITATIVE"), "adj_quantitative_lemma", False
        if w in MODAL_ADJ:
            return ("CAN", "ADJ_MODAL"), "adj_modal_lemma", False
        if adj.get("adj.pert"):
            return ("BIG", "ADJ_DESCRIPTIVE"), "adj_relational_unreachable", True
        return ("BIG", "ADJ_DESCRIPTIVE"), "adj_descriptive_default", True

    def _adv(self, w, e):
        if w in DEGREE_ADV:
            return ("VERY" if w != "more" else "MORE", "ADV_DEGREE"), "adv_degree_lemma", False
        if w in TEMPORAL_ADV:
            prim = {"now": "NOW", "before": "BEFORE", "after": "AFTER"}.get(w, "NOW")
            return (prim, "ADV_TEMPORAL"), "adv_temporal_lemma", w not in ("now", "before", "after")
        if w in SPATIAL_ADV:
            prim = {"here": "HERE", "near": "NEAR", "far": "FAR"}.get(w, "HERE")
            return (prim, "ADV_SPATIAL"), "adv_spatial_lemma", w not in ("here", "near", "far")
        if w in FREQ_ADV:
            return ("MAYBE", "ADV_MODAL"), "adv_frequency_unreachable", True
        if w in EVAL_ADV:
            return ("GOOD" if w != "badly" else "BAD", "ADV_EVALUATIVE"), "adv_evaluative_lemma", False
        if w in MODAL_ADV:
            return ("NOT" if w == "not" else "MAYBE", "ADV_MODAL"), "adv_modal_lemma", False
        return ("MAYBE", "ADV_MODAL"), "adv_manner_unreachable", True

    # ------------------------------------------------------------------
    def _fallback(self, w, phonemes):
        """The original heuristic, preserved verbatim in spirit as the
        declared last resort for words with no classification evidence."""
        if len(w) <= 3:
            return ("THING", "NOUN_CONCRETE")
        first = (phonemes[0].rstrip("012") if phonemes else "")
        if first in {"R", "L", "S", "P", "B", "T", "D"}:
            return ("MOVE", "VERB_ACTION")
        if first in {"F", "V", "TH", "N", "M"}:
            return ("THINK", "VERB_MENTAL")
        return ("THING", "NOUN_CONCRETE")


if __name__ == "__main__":
    clf = LexiconL1()
    print("reachable subpos:", len(REACHABLE_SUBPOS), "of", len(_matrix.L1_SUBPOS))
    print("unreachable:", sorted(set(_matrix.L1_SUBPOS) - REACHABLE_SUBPOS))
    for word in ["dog", "paris", "run", "quickly", "happy", "water", "idea",
                 "say", "think", "the", "of", "very", "freedom", "photosynthesis"]:
        pair, prov = clf.assign_with_provenance(word, ["X"])
        print(f"  {word:16s} -> {pair}  {prov}")
