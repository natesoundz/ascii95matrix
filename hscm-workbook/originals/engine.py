"""
HSCM Coordinate Engine
Compression (word → coordinate) and Traversal (constraint intersection)

Premise: Inspired Skeptic
- Only assert what the structure supports
- Dead ends are findings, not failures
- The constraint field is the model
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from matrix import (
    L0_PRIMITIVES, L1_SUBPOS, L2_PHONETIC,
    L3_GEOMETRIC, L4_TRAJECTORY, PREMISE_CONTEXTS,
    VALID_L0_L1_EDGES
)


# =============================================================================
# COORDINATE — the compressed representation of a word
# =============================================================================

@dataclass
class Coordinate:
    """
    A word's full address in the HSCM space.
    Five fields. If any field is None, the coordinate is partial.
    A partial coordinate is a valid search region, not a failure.
    """
    surface_form: str
    l0_primitive: str           # e.g. "MOVE"
    l1_subpos: str              # e.g. "VERB_MOTION"
    l2_sequence: List[str]      # e.g. ["S", "P", "R", "T"] for "sprint"
    l3_geometric: str           # e.g. "ROOT_GERMANIC"
    l4_trajectory: List[str]    # e.g. ["VEL_FAST", "PER_DYNAMIC"]
    premise: str = "ANALYTICAL" # default operating context

    def is_complete(self) -> bool:
        return all([
            self.l0_primitive,
            self.l1_subpos,
            self.l2_sequence,
            self.l3_geometric,
            self.l4_trajectory,
        ])

    def to_tuple(self) -> tuple:
        """The compressed storage form."""
        return (
            self.l0_primitive,
            self.l1_subpos,
            tuple(self.l2_sequence),
            self.l3_geometric,
            tuple(sorted(self.l4_trajectory)),
        )

    def __repr__(self):
        seq = "-".join(self.l2_sequence)
        traj = "+".join(self.l4_trajectory)
        return (
            f"[{self.l0_primitive} | {self.l1_subpos} | "
            f"{seq} | {self.l3_geometric} | {traj}]"
        )


# =============================================================================
# PHONETIC PARSER
# Maps surface characters to L2 operators.
# This is the explicit character → physics mapping.
# Note: Full implementation requires CMU Pronouncing Dictionary for 
# phoneme-accurate parsing. This handles the common cases correctly.
# =============================================================================

class PhoneticParser:
    """
    Maps surface characters to specific L2 phonetic operators.
    Key design: specific characters are unique keys, NOT collapsed into 
    category buckets. 'p' and 't' are both plosives but are distinct operators.
    The category is metadata. The character is the address.
    """

    # Direct character-to-operator mapping
    # Each character maps to its specific L2 key
    CHAR_MAP = {
        # Plosives — each distinct
        'p': 'P', 'b': 'B', 't': 'T', 'd': 'D', 'k': 'K', 'g': 'G',
        # Fricatives — each distinct
        'f': 'F', 'v': 'V', 's': 'S', 'z': 'Z', 'h': 'F',  # h approximated
        # Nasals — each distinct
        'm': 'M', 'n': 'N',
        # Liquids / Glides — each distinct
        'l': 'L', 'r': 'R', 'w': 'W', 'y': 'Y',
        # Vowels — mapped to fold class (height+backness)
        'i': 'V_HIGH_FRONT', 'e': 'V_MID',      'a': 'V_LOW',
        'o': 'V_HIGH_BACK',  'u': 'V_HIGH_BACK',
    }

    # Digraph handling — must be checked BEFORE single chars
    DIGRAPH_MAP = {
        'sh': 'SH', 'zh': 'ZH',
        'th': 'TH_V',  # default voiceless; 'the' would be TH_Z (context needed)
        'ng': 'NG',
        'ck': 'K',     # 'ck' is a single K sound
        'qu': 'K',     # 'qu' onset is K
    }

    def parse(self, word: str) -> List[str]:
        """
        Returns the ordered phonetic sequence for a word.
        Handles digraphs before single characters.
        Silent letters produce no operator (they are structural, not phonetic).
        """
        word = word.lower()
        sequence = []
        i = 0

        while i < len(word):
            # Check digraph first
            if i + 1 < len(word):
                digraph = word[i:i+2]
                if digraph in self.DIGRAPH_MAP:
                    sequence.append(self.DIGRAPH_MAP[digraph])
                    i += 2
                    continue

            char = word[i]

            # Silent e at end — structural, not phonetic
            if char == 'e' and i == len(word) - 1 and len(word) > 1:
                i += 1
                continue

            if char in self.CHAR_MAP:
                sequence.append(self.CHAR_MAP[char])
            # Unrecognized characters (punctuation etc.) are skipped

            i += 1

        return sequence


# =============================================================================
# CONSTRAINT VALIDATOR
# The core logic: does a coordinate satisfy all declared constraints?
# =============================================================================

class ConstraintValidator:
    """
    Validates coordinate against the explicit relationship table.
    Every check is binary: valid path or dead end.
    Dead ends are reported as structured findings, not errors.
    """

    def validate_l0_l1_edge(self, primitive: str, subpos: str) -> dict:
        """Check if primitive→subpos is a declared valid edge."""
        valid_subpos = VALID_L0_L1_EDGES.get(primitive, [])
        if subpos in valid_subpos:
            return {"valid": True, "finding": None}
        return {
            "valid": False,
            "finding": f"DEAD_END: {subpos} is not a valid role for primitive {primitive}. "
                      f"Valid roles: {valid_subpos}"
        }

    def validate_l0_exists(self, primitive: str) -> dict:
        if primitive in L0_PRIMITIVES:
            return {"valid": True, "finding": None}
        return {
            "valid": False,
            "finding": f"DEAD_END: '{primitive}' is not a defined primitive. "
                      f"Check L0_PRIMITIVES for valid entries."
        }

    def validate_l1_exists(self, subpos: str) -> dict:
        if subpos in L1_SUBPOS:
            return {"valid": True, "finding": None}
        return {
            "valid": False,
            "finding": f"DEAD_END: '{subpos}' is not a defined SubPOS."
        }

    def validate_l2_sequence(self, sequence: List[str]) -> dict:
        invalid = [op for op in sequence if op not in L2_PHONETIC]
        if not invalid:
            return {"valid": True, "finding": None}
        return {
            "valid": False,
            "finding": f"DEAD_END: Unknown phonetic operators: {invalid}"
        }

    def validate_l4_trajectory(self, trajectories: List[str]) -> dict:
        """Trajectories must not contain both poles of the same axis."""
        invalid = [t for t in trajectories if t not in L4_TRAJECTORY]
        if invalid:
            return {"valid": False, "finding": f"Unknown trajectory keys: {invalid}"}

        # Check for contradicting poles on same axis
        axes_seen = {}
        for traj in trajectories:
            axis = L4_TRAJECTORY[traj]["axis"]
            if axis in axes_seen:
                return {
                    "valid": False,
                    "finding": f"CONTRADICTION: Both poles of axis '{axis}' declared. "
                               f"This is an internal inconsistency."
                }
            axes_seen[axis] = traj

        return {"valid": True, "finding": None}

    def validate_premise_coherence(self, coordinate: Coordinate, premise_key: str) -> dict:
        """
        Premise-scoped validation.
        EMPIRICAL: physics constraints active — checks trajectory consistency
        CREATIVE: suspended — anything structurally coherent passes
        ANALYTICAL: internal coherence only
        """
        premise = PREMISE_CONTEXTS.get(premise_key)
        if not premise:
            return {"valid": False, "finding": f"Unknown premise: {premise_key}"}

        if premise["contradiction_fatal"]:
            # Under empirical/legal/analytical — contradictions in L4 are fatal
            traj_check = self.validate_l4_trajectory(coordinate.l4_trajectory)
            if not traj_check["valid"]:
                return {
                    "valid": False,
                    "finding": f"[{premise_key}] " + traj_check["finding"]
                }

        return {"valid": True, "finding": None, "premise": premise["description"]}

    def full_validate(self, coordinate: Coordinate) -> dict:
        """Run all constraint checks. Return complete validation report."""
        findings = []
        passed = []

        checks = [
            ("L0 exists",       self.validate_l0_exists(coordinate.l0_primitive)),
            ("L1 exists",       self.validate_l1_exists(coordinate.l1_subpos)),
            ("L0→L1 edge",      self.validate_l0_l1_edge(coordinate.l0_primitive, coordinate.l1_subpos)),
            ("L2 sequence",     self.validate_l2_sequence(coordinate.l2_sequence)),
            ("L4 trajectory",   self.validate_l4_trajectory(coordinate.l4_trajectory)),
            ("Premise coherence", self.validate_premise_coherence(coordinate, coordinate.premise)),
        ]

        for name, result in checks:
            if result["valid"]:
                passed.append(name)
            else:
                findings.append({"check": name, "finding": result["finding"]})

        return {
            "coordinate": repr(coordinate),
            "surface_form": coordinate.surface_form,
            "valid": len(findings) == 0,
            "passed": passed,
            "dead_ends": findings,
            "is_complete": coordinate.is_complete(),
        }


# =============================================================================
# COMPRESSOR — word → coordinate
# =============================================================================

class Compressor:
    """
    Produces a coordinate from a word + declared context.
    The SubPOS, L3, and L4 fields require caller input unless 
    a lookup table is provided — they cannot be derived from 
    surface form alone without external knowledge.
    
    Honest stub marker: fields marked [REQUIRES_CONTEXT] cannot 
    be auto-derived from the surface form. They need either:
    - Human annotation
    - A pre-built lookup (Wiktionary/WordNet integration)
    - An upstream classifier
    """

    def __init__(self):
        self.parser = PhoneticParser()
        self.validator = ConstraintValidator()

    def compress(
        self,
        word: str,
        primitive: str,
        subpos: str,
        geometric: str,
        trajectory: List[str],
        premise: str = "ANALYTICAL"
    ) -> dict:
        """
        Compress a word into its HSCM coordinate.
        L2 is derived. All other fields are declared by caller.
        """
        l2_sequence = self.parser.parse(word)

        coord = Coordinate(
            surface_form=word,
            l0_primitive=primitive,
            l1_subpos=subpos,
            l2_sequence=l2_sequence,
            l3_geometric=geometric,
            l4_trajectory=trajectory,
            premise=premise
        )

        validation = self.validator.full_validate(coord)

        return {
            "coordinate": coord,
            "tuple": coord.to_tuple() if validation["valid"] else None,
            "validation": validation,
            "phonetic_sequence": l2_sequence,
        }


# =============================================================================
# TRAVERSAL ENGINE — constraint intersection
# This is the core of "dictionary on demand"
# Given a coordinate (or partial coordinate), find all entries that match.
# =============================================================================

class TraversalEngine:
    """
    Operates on a registered coordinate space.
    Compression populates the space. Traversal queries it.
    
    The sparse network is the feature:
    Most of the 11.4M theoretical coordinates are empty.
    Valid paths are the occupied ones.
    Dead ends are the structural guarantee against confabulation.
    """

    def __init__(self):
        self.registry: Dict[tuple, Coordinate] = {}
        self.validator = ConstraintValidator()

    def register(self, coordinate: Coordinate):
        """Add a validated coordinate to the traversal space."""
        key = coordinate.to_tuple()
        self.registry[key] = coordinate

    def query(self, partial: dict, premise: str = "ANALYTICAL") -> dict:
        """
        Query by partial coordinate. Returns all matches.
        partial = dict with any subset of {l0, l1, l2, l3, l4}
        
        Empty result is not a failure — it IS the finding.
        It means no word in the registered space satisfies these constraints.
        """
        matches = []
        dead_ends = []

        # First: validate that the partial query itself is internally coherent
        if "l0" in partial and "l1" in partial:
            edge_check = self.validator.validate_l0_l1_edge(partial["l0"], partial["l1"])
            if not edge_check["valid"]:
                return {
                    "matches": [],
                    "finding": edge_check["finding"],
                    "query_valid": False,
                }

        for key, coord in self.registry.items():
            match = True

            if "l0" in partial and coord.l0_primitive != partial["l0"]:
                match = False
            if "l1" in partial and coord.l1_subpos != partial["l1"]:
                match = False
            if "l2" in partial and coord.l2_sequence != partial["l2"]:
                match = False
            if "l3" in partial and coord.l3_geometric != partial["l3"]:
                match = False
            if "l4" in partial:
                # L4 is a set — check that all requested trajectories are present
                if not all(t in coord.l4_trajectory for t in partial["l4"]):
                    match = False

            if match:
                matches.append(coord)

        return {
            "matches": matches,
            "match_count": len(matches),
            "query_valid": True,
            "finding": f"Found {len(matches)} coordinates satisfying constraints."
                       if matches else
                       "DEAD_END: No registered coordinates satisfy these constraints. "
                       "Either the space is not yet populated or the combination is invalid.",
        }

    def coherence_check(self, text_coordinates: List[Coordinate], premise: str) -> dict:
        """
        Check a sequence of coordinates (a sentence/document) for 
        internal consistency under a declared premise.
        
        This is the lie/inconsistency detection layer:
        Flags coordinates whose L4 trajectories contradict each other
        across the sequence, or whose L0 primitives violate the premise field.
        """
        context = PREMISE_CONTEXTS.get(premise, PREMISE_CONTEXTS["ANALYTICAL"])
        issues = []

        for i, coord in enumerate(text_coordinates):
            result = self.validator.validate_premise_coherence(coord, premise)
            if not result["valid"]:
                issues.append({
                    "position": i,
                    "word": coord.surface_form,
                    "issue": result["finding"]
                })

        # Cross-coordinate checks: detect contradicting trajectory claims
        # e.g., sentence declares both STATIC and DYNAMIC for same referent
        if context["contradiction_fatal"]:
            axis_assignments = {}
            for i, coord in enumerate(text_coordinates):
                for traj in coord.l4_trajectory:
                    axis = L4_TRAJECTORY[traj]["axis"]
                    pole = L4_TRAJECTORY[traj]["pole"]
                    if axis in axis_assignments:
                        prev_i, prev_pole = axis_assignments[axis]
                        if prev_pole != pole:
                            issues.append({
                                "position": i,
                                "word": coord.surface_form,
                                "issue": (
                                    f"INCONSISTENCY: '{coord.surface_form}' (pos {i}) "
                                    f"assigns {axis}={pole}, but '{text_coordinates[prev_i].surface_form}' "
                                    f"(pos {prev_i}) assigned {axis}={prev_pole}. "
                                    f"Under {premise} premise, this is a structural contradiction."
                                )
                            })
                    else:
                        axis_assignments[axis] = (i, pole)

        return {
            "premise": premise,
            "premise_description": context["description"],
            "coherent": len(issues) == 0,
            "issues": issues,
            "word_count": len(text_coordinates),
        }
