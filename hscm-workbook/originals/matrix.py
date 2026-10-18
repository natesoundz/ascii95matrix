"""
HSCM Matrix — The 162 KPI Definition Layer
Hierarchical Semantic Coordinate Matrix

This module defines the complete constraint field.
Nothing is computed here. This is the declared topology.
All relationships are explicit. All others are zero.
"""

# =============================================================================
# L0: SEMANTIC PRIMITIVES (65)
# Wierzbicka NSM-derived. These are the irreducible semantic atoms.
# Every word in the language maps to exactly one of these.
# =============================================================================

L0_PRIMITIVES = {
    # SUBSTANTIVES
    "THING":     {"id": 0,  "category": "SUBSTANTIVE"},
    "SOMEBODY":  {"id": 1,  "category": "SUBSTANTIVE"},
    "PEOPLE":    {"id": 2,  "category": "SUBSTANTIVE"},
    "BODY":      {"id": 3,  "category": "SUBSTANTIVE"},
    "KIND":      {"id": 4,  "category": "SUBSTANTIVE"},
    "PART":      {"id": 5,  "category": "SUBSTANTIVE"},

    # DETERMINERS
    "THIS":      {"id": 6,  "category": "DETERMINER"},
    "THE_SAME":  {"id": 7,  "category": "DETERMINER"},
    "OTHER":     {"id": 8,  "category": "DETERMINER"},

    # QUANTIFIERS
    "ONE":       {"id": 9,  "category": "QUANTIFIER"},
    "TWO":       {"id": 10, "category": "QUANTIFIER"},
    "SOME":      {"id": 11, "category": "QUANTIFIER"},
    "ALL":       {"id": 12, "category": "QUANTIFIER"},
    "MUCH":      {"id": 13, "category": "QUANTIFIER"},
    "MANY":      {"id": 14, "category": "QUANTIFIER"},

    # EVALUATORS
    "GOOD":      {"id": 15, "category": "EVALUATOR"},
    "BAD":       {"id": 16, "category": "EVALUATOR"},

    # DESCRIPTORS
    "BIG":       {"id": 17, "category": "DESCRIPTOR"},
    "SMALL":     {"id": 18, "category": "DESCRIPTOR"},

    # MENTAL PREDICATES
    "THINK":     {"id": 19, "category": "MENTAL"},
    "KNOW":      {"id": 20, "category": "MENTAL"},
    "WANT":      {"id": 21, "category": "MENTAL"},
    "FEEL":      {"id": 22, "category": "MENTAL"},
    "SEE":       {"id": 23, "category": "MENTAL"},
    "HEAR":      {"id": 24, "category": "MENTAL"},

    # SPEECH
    "SAY":       {"id": 25, "category": "SPEECH"},
    "WORDS":     {"id": 26, "category": "SPEECH"},
    "TRUE":      {"id": 27, "category": "SPEECH"},

    # ACTIONS / EVENTS
    "DO":        {"id": 28, "category": "ACTION"},
    "HAPPEN":    {"id": 29, "category": "ACTION"},
    "MOVE":      {"id": 30, "category": "ACTION"},
    "TOUCH":     {"id": 31, "category": "ACTION"},

    # EXISTENCE / POSSESSION
    "BE":        {"id": 32, "category": "EXISTENCE"},
    "THERE_IS":  {"id": 33, "category": "EXISTENCE"},
    "HAVE":      {"id": 34, "category": "EXISTENCE"},

    # LIFE / DEATH
    "LIVE":      {"id": 35, "category": "LIFE"},
    "DIE":       {"id": 36, "category": "LIFE"},

    # TIME
    "WHEN":      {"id": 37, "category": "TIME"},
    "NOW":       {"id": 38, "category": "TIME"},
    "BEFORE":    {"id": 39, "category": "TIME"},
    "AFTER":     {"id": 40, "category": "TIME"},
    "A_LONG_TIME":{"id": 41, "category": "TIME"},
    "A_SHORT_TIME":{"id": 42, "category": "TIME"},
    "FOR_SOME_TIME":{"id": 43,"category": "TIME"},
    "MOMENT":    {"id": 44, "category": "TIME"},

    # SPACE
    "WHERE":     {"id": 45, "category": "SPACE"},
    "HERE":      {"id": 46, "category": "SPACE"},
    "ABOVE":     {"id": 47, "category": "SPACE"},
    "BELOW":     {"id": 48, "category": "SPACE"},
    "FAR":       {"id": 49, "category": "SPACE"},
    "NEAR":      {"id": 50, "category": "SPACE"},
    "SIDE":      {"id": 51, "category": "SPACE"},
    "INSIDE":    {"id": 52, "category": "SPACE"},
    "OUTSIDE":   {"id": 53, "category": "SPACE"},

    # LOGICAL RELATIONS
    "NOT":       {"id": 54, "category": "LOGICAL"},
    "MAYBE":     {"id": 55, "category": "LOGICAL"},

    # AUGMENTORS
    "VERY":      {"id": 56, "category": "AUGMENTOR"},
    "MORE":      {"id": 57, "category": "AUGMENTOR"},

    # TAXONOMY
    "LIKE":      {"id": 58, "category": "TAXONOMY"},
    "AS":        {"id": 59, "category": "TAXONOMY"},

    # SIMILARITY
    "IF":        {"id": 60, "category": "SIMILARITY"},
    "BECAUSE":   {"id": 61, "category": "SIMILARITY"},

    # ADDITIONAL (to reach 65)
    "CAN":       {"id": 62, "category": "MODAL"},
    "SHOULD":    {"id": 63, "category": "MODAL"},
    "BECOME":    {"id": 64, "category": "ACTION"},
}

# =============================================================================
# L1: SUB-CATEGORY PARTS OF SPEECH (45)
# The functional role of the word within the primitive's domain.
# =============================================================================

L1_SUBPOS = {
    # NOUN SUB-CATEGORIES (11)
    "NOUN_CONCRETE":    {"id": 0,  "parent": "NOUN"},
    "NOUN_ABSTRACT":    {"id": 1,  "parent": "NOUN"},
    "NOUN_COLLECTIVE":  {"id": 2,  "parent": "NOUN"},
    "NOUN_MASS":        {"id": 3,  "parent": "NOUN"},
    "NOUN_PROPER":      {"id": 4,  "parent": "NOUN"},
    "NOUN_ANIMATE":     {"id": 5,  "parent": "NOUN"},
    "NOUN_INANIMATE":   {"id": 6,  "parent": "NOUN"},
    "NOUN_PROCESS":     {"id": 7,  "parent": "NOUN"},
    "NOUN_STATE":       {"id": 8,  "parent": "NOUN"},
    "NOUN_EVENT":       {"id": 9,  "parent": "NOUN"},
    "NOUN_ROLE":        {"id": 10, "parent": "NOUN"},

    # VERB SUB-CATEGORIES (11)
    "VERB_ACTION":      {"id": 11, "parent": "VERB"},
    "VERB_STATE":       {"id": 12, "parent": "VERB"},
    "VERB_PROCESS":     {"id": 13, "parent": "VERB"},
    "VERB_CAUSATIVE":   {"id": 14, "parent": "VERB"},
    "VERB_MENTAL":      {"id": 15, "parent": "VERB"},
    "VERB_SPEECH":      {"id": 16, "parent": "VERB"},
    "VERB_MOTION":      {"id": 17, "parent": "VERB"},
    "VERB_PERCEPTION":  {"id": 18, "parent": "VERB"},
    "VERB_EXISTENCE":   {"id": 19, "parent": "VERB"},
    "VERB_POSSESSION":  {"id": 20, "parent": "VERB"},
    "VERB_CHANGE":      {"id": 21, "parent": "VERB"},

    # ADJECTIVE SUB-CATEGORIES (8)
    "ADJ_EVALUATIVE":   {"id": 22, "parent": "ADJ"},
    "ADJ_DESCRIPTIVE":  {"id": 23, "parent": "ADJ"},
    "ADJ_RELATIONAL":   {"id": 24, "parent": "ADJ"},
    "ADJ_TEMPORAL":     {"id": 25, "parent": "ADJ"},
    "ADJ_SPATIAL":      {"id": 26, "parent": "ADJ"},
    "ADJ_QUANTITATIVE": {"id": 27, "parent": "ADJ"},
    "ADJ_AFFECTIVE":    {"id": 28, "parent": "ADJ"},
    "ADJ_MODAL":        {"id": 29, "parent": "ADJ"},

    # ADVERB SUB-CATEGORIES (7)
    "ADV_MANNER":       {"id": 30, "parent": "ADV"},
    "ADV_DEGREE":       {"id": 31, "parent": "ADV"},
    "ADV_TEMPORAL":     {"id": 32, "parent": "ADV"},
    "ADV_SPATIAL":      {"id": 33, "parent": "ADV"},
    "ADV_FREQUENCY":    {"id": 34, "parent": "ADV"},
    "ADV_MODAL":        {"id": 35, "parent": "ADV"},
    "ADV_EVALUATIVE":   {"id": 36, "parent": "ADV"},

    # STRUCTURAL / OTHER (8)
    "PRONOUN_PERSONAL": {"id": 37, "parent": "STRUCTURAL"},
    "PRONOUN_RELATIVE": {"id": 38, "parent": "STRUCTURAL"},
    "PREPOSITION_SPATIAL":{"id": 39,"parent": "STRUCTURAL"},
    "PREPOSITION_TEMPORAL":{"id": 40,"parent":"STRUCTURAL"},
    "CONJUNCTION_COORD":{"id": 41, "parent": "STRUCTURAL"},
    "CONJUNCTION_SUB":  {"id": 42, "parent": "STRUCTURAL"},
    "DETERMINER_DEF":   {"id": 43, "parent": "STRUCTURAL"},
    "PARTICLE":         {"id": 44, "parent": "STRUCTURAL"},
}

# =============================================================================
# L2: PHONETIC OPERATORS (28)
# The specific mouth physics. NOT categories — specific characters.
# p != t != k even though all are plosives.
# =============================================================================

L2_PHONETIC = {
    # PLOSIVES (6) — sudden state change, impact, boundary
    "P": {"id": 0,  "class": "PLOSIVE",   "voiced": False, "place": "bilabial"},
    "B": {"id": 1,  "class": "PLOSIVE",   "voiced": True,  "place": "bilabial"},
    "T": {"id": 2,  "class": "PLOSIVE",   "voiced": False, "place": "alveolar"},
    "D": {"id": 3,  "class": "PLOSIVE",   "voiced": True,  "place": "alveolar"},
    "K": {"id": 4,  "class": "PLOSIVE",   "voiced": False, "place": "velar"},
    "G": {"id": 5,  "class": "PLOSIVE",   "voiced": True,  "place": "velar"},

    # FRICATIVES (8) — continuance, sustained awareness, friction
    "F": {"id": 6,  "class": "FRICATIVE", "voiced": False, "place": "labiodental"},
    "V": {"id": 7,  "class": "FRICATIVE", "voiced": True,  "place": "labiodental"},
    "S": {"id": 8,  "class": "FRICATIVE", "voiced": False, "place": "alveolar"},
    "Z": {"id": 9,  "class": "FRICATIVE", "voiced": True,  "place": "alveolar"},
    "SH":{"id": 10, "class": "FRICATIVE", "voiced": False, "place": "postalveolar"},
    "ZH":{"id": 11, "class": "FRICATIVE", "voiced": True,  "place": "postalveolar"},
    "TH_V":{"id": 12,"class":"FRICATIVE", "voiced": False, "place": "dental"},
    "TH_Z":{"id": 13,"class":"FRICATIVE", "voiced": True,  "place": "dental"},

    # NASALS (3) — internalized identity, resonance, presence
    "M": {"id": 14, "class": "NASAL",     "voiced": True,  "place": "bilabial"},
    "N": {"id": 15, "class": "NASAL",     "voiced": True,  "place": "alveolar"},
    "NG":{"id": 16, "class": "NASAL",     "voiced": True,  "place": "velar"},

    # LIQUIDS / GLIDES (4) — directional flow, reach, spatial relationship
    "L": {"id": 17, "class": "LIQUID",    "voiced": True,  "place": "alveolar"},
    "R": {"id": 18, "class": "LIQUID",    "voiced": True,  "place": "alveolar"},
    "W": {"id": 19, "class": "GLIDE",     "voiced": True,  "place": "bilabial"},
    "Y": {"id": 20, "class": "GLIDE",     "voiced": True,  "place": "palatal"},

    # VOWEL FOLDS (4 classes, but each character is distinct)
    # High front: i, ee
    "V_HIGH_FRONT": {"id": 21, "class": "VOWEL", "height": "high",  "backness": "front"},
    # Mid: e, a (schwa region)
    "V_MID":        {"id": 22, "class": "VOWEL", "height": "mid",   "backness": "central"},
    # Low: ae, ah
    "V_LOW":        {"id": 23, "class": "VOWEL", "height": "low",   "backness": "front"},
    # High back / round: oo, oh
    "V_HIGH_BACK":  {"id": 24, "class": "VOWEL", "height": "high",  "backness": "back"},

    # TENSION STATES (3) — prosodic and coarticulatory behavior
    "ASPIRATED":    {"id": 25, "class": "TENSION", "effect": "burst_release"},
    "GEMINATE":     {"id": 26, "class": "TENSION", "effect": "doubled_hold"},
    "REDUCED":      {"id": 27, "class": "TENSION", "effect": "schwa_reduction"},
}

# =============================================================================
# L3: GEOMETRIC KEYS (14)
# Lineage and topological properties. These are structural metadata.
# =============================================================================

L3_GEOMETRIC = {
    # ETYMOLOGICAL ROOTS (6)
    "ROOT_LATIN":    {"id": 0,  "type": "ETYMOLOGY", "origin": "Latin"},
    "ROOT_GREEK":    {"id": 1,  "type": "ETYMOLOGY", "origin": "Greek"},
    "ROOT_GERMANIC": {"id": 2,  "type": "ETYMOLOGY", "origin": "Germanic"},
    "ROOT_FRENCH":   {"id": 3,  "type": "ETYMOLOGY", "origin": "Old French"},
    "ROOT_NORSE":    {"id": 4,  "type": "ETYMOLOGY", "origin": "Old Norse"},
    "ROOT_NATIVE":   {"id": 5,  "type": "ETYMOLOGY", "origin": "Native/Other"},

    # TENSION GRADIENTS (2)
    "GRADIENT_RISING":  {"id": 6,  "type": "TENSION", "direction": "ascending"},
    "GRADIENT_FALLING": {"id": 7,  "type": "TENSION", "direction": "descending"},

    # CHARACTER TOPOLOGY (3)
    "TOPO_OPEN":    {"id": 8,  "type": "TOPOLOGY", "form": "open_syllable"},
    "TOPO_CLOSED":  {"id": 9,  "type": "TOPOLOGY", "form": "closed_syllable"},
    "TOPO_CLUSTER": {"id": 10, "type": "TOPOLOGY", "form": "consonant_cluster"},

    # SYLLABIC WEIGHT (3)
    "SYL_LIGHT":    {"id": 11, "type": "SYLLABIC", "weight": "light"},
    "SYL_HEAVY":    {"id": 12, "type": "SYLLABIC", "weight": "heavy"},
    "SYL_SUPERHEAVY":{"id": 13,"type": "SYLLABIC", "weight": "superheavy"},
}

# =============================================================================
# L4: IMPLICIT TRAJECTORIES (10)
# The pragmatic/affective vector of the word within its context.
# =============================================================================

L4_TRAJECTORY = {
    # VELOCITY
    "VEL_FAST":   {"id": 0, "axis": "VELOCITY",     "pole": "fast"},
    "VEL_SLOW":   {"id": 1, "axis": "VELOCITY",     "pole": "slow"},

    # MAGNITUDE
    "MAG_LARGE":  {"id": 2, "axis": "MAGNITUDE",    "pole": "large"},
    "MAG_SMALL":  {"id": 3, "axis": "MAGNITUDE",    "pole": "small"},

    # DIRECTIONALITY
    "DIR_OUT":    {"id": 4, "axis": "DIRECTION",    "pole": "outward"},
    "DIR_IN":     {"id": 5, "axis": "DIRECTION",    "pole": "inward"},

    # AFFECTIVE BIAS
    "AFF_POS":    {"id": 6, "axis": "AFFECTIVE",    "pole": "positive"},
    "AFF_NEG":    {"id": 7, "axis": "AFFECTIVE",    "pole": "negative"},

    # PERSISTENCE
    "PER_STATIC": {"id": 8, "axis": "PERSISTENCE",  "pole": "static"},
    "PER_DYNAMIC":{"id": 9, "axis": "PERSISTENCE",  "pole": "dynamic"},
}

# =============================================================================
# PREMISE FIELD
# The identity layer. Declares intent before any traversal begins.
# This gates which validation rules are active.
# =============================================================================

PREMISE_CONTEXTS = {
    "EMPIRICAL":    {
        "id": 0,
        "description": "Factual claim. Physics constraints active. External coherence required.",
        "physics_check": True,
        "metaphor_allowed": False,
        "contradiction_fatal": True,
    },
    "CREATIVE":     {
        "id": 1,
        "description": "Metaphorical or fictional. Physics constraints suspended.",
        "physics_check": False,
        "metaphor_allowed": True,
        "contradiction_fatal": False,
    },
    "LEGAL":        {
        "id": 2,
        "description": "Formal assertion. Internal consistency required. Ambiguity is failure.",
        "physics_check": False,
        "metaphor_allowed": False,
        "contradiction_fatal": True,
    },
    "CONVERSATIONAL":{
        "id": 3,
        "description": "Pragmatic exchange. Loose constraints. Context-completion active.",
        "physics_check": False,
        "metaphor_allowed": True,
        "contradiction_fatal": False,
    },
    "ANALYTICAL":   {
        "id": 4,
        "description": "Structural analysis. Internal coherence required. No external reference.",
        "physics_check": False,
        "metaphor_allowed": False,
        "contradiction_fatal": True,
    },
}

# =============================================================================
# VALID PRIMITIVE → SUBPOS MAPPINGS
# Explicit relationship table. Everything not listed here is zero.
# =============================================================================

VALID_L0_L1_EDGES = {
    "MOVE":    ["VERB_MOTION", "VERB_ACTION", "NOUN_PROCESS", "NOUN_EVENT"],
    "DO":      ["VERB_ACTION", "VERB_CAUSATIVE", "NOUN_EVENT"],
    "HAPPEN":  ["VERB_PROCESS", "VERB_STATE", "NOUN_EVENT"],
    "THINK":   ["VERB_MENTAL", "NOUN_PROCESS", "NOUN_STATE"],
    "KNOW":    ["VERB_MENTAL", "VERB_STATE", "NOUN_STATE", "NOUN_ABSTRACT"],
    "WANT":    ["VERB_MENTAL", "VERB_STATE", "NOUN_STATE"],
    "FEEL":    ["VERB_MENTAL", "VERB_PERCEPTION", "NOUN_STATE", "NOUN_ABSTRACT"],
    "SEE":     ["VERB_PERCEPTION", "VERB_MENTAL", "NOUN_EVENT"],
    "HEAR":    ["VERB_PERCEPTION", "NOUN_EVENT"],
    "SAY":     ["VERB_SPEECH", "NOUN_EVENT", "NOUN_PROCESS"],
    "GOOD":    ["ADJ_EVALUATIVE", "ADJ_AFFECTIVE", "ADV_EVALUATIVE", "NOUN_ABSTRACT"],
    "BAD":     ["ADJ_EVALUATIVE", "ADJ_AFFECTIVE", "ADV_EVALUATIVE", "NOUN_ABSTRACT"],
    "BIG":     ["ADJ_DESCRIPTIVE", "ADJ_QUANTITATIVE"],
    "SMALL":   ["ADJ_DESCRIPTIVE", "ADJ_QUANTITATIVE"],
    "THING":   ["NOUN_CONCRETE", "NOUN_ABSTRACT", "NOUN_INANIMATE"],
    "SOMEBODY":["NOUN_ANIMATE", "PRONOUN_PERSONAL"],
    "HAVE":    ["VERB_POSSESSION", "VERB_EXISTENCE"],
    "BE":      ["VERB_EXISTENCE", "VERB_STATE"],
    "BECOME":  ["VERB_CHANGE", "VERB_PROCESS"],
    "LIVE":    ["VERB_STATE", "VERB_ACTION", "VERB_EXISTENCE"],
    "DIE":     ["VERB_CHANGE", "VERB_PROCESS", "NOUN_EVENT"],
    "TOUCH":   ["VERB_ACTION", "VERB_PERCEPTION", "NOUN_EVENT"],
    "VERY":    ["ADV_DEGREE"],
    "MORE":    ["ADV_DEGREE", "ADJ_QUANTITATIVE"],
    "NOT":     ["PARTICLE", "ADV_MODAL"],
    "MAYBE":   ["ADV_MODAL", "PARTICLE"],
    "HERE":    ["ADV_SPATIAL", "PREPOSITION_SPATIAL"],
    "NEAR":    ["ADJ_SPATIAL", "ADV_SPATIAL", "PREPOSITION_SPATIAL"],
    "FAR":     ["ADJ_SPATIAL", "ADV_SPATIAL"],
    "NOW":     ["ADV_TEMPORAL"],
    "BEFORE":  ["ADV_TEMPORAL", "PREPOSITION_TEMPORAL", "CONJUNCTION_SUB"],
    "AFTER":   ["ADV_TEMPORAL", "PREPOSITION_TEMPORAL", "CONJUNCTION_SUB"],
    "WORDS":   ["NOUN_ABSTRACT", "NOUN_CONCRETE"],
    "TRUE":    ["ADJ_EVALUATIVE", "NOUN_ABSTRACT"],
    "LIKE":    ["VERB_MENTAL", "PREPOSITION_SPATIAL", "CONJUNCTION_COORD"],
    "BECAUSE": ["CONJUNCTION_SUB"],
    "IF":      ["CONJUNCTION_SUB"],
    "CAN":     ["ADJ_MODAL", "ADV_MODAL"],
    "SHOULD":  ["ADJ_MODAL", "ADV_MODAL"],
}
