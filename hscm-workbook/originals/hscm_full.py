"""
HSCM Full System — Weekend Experiment Package
=============================================

Components:
  1. Context Shift Loop     — fine-tuning via premise refinement
  2. Shift Log              — every adjustment recorded as S-expression
  3. Rollback               — one call to undo any shift
  4. RAG Layer              — retrieval-augmented generation over the manifold
  5. Bytecode Embedding     — precompiled Python stored in LISP lists

Run:  python hscm_full.py
      python hscm_full.py --rag "what words mean slow deliberate motion"
      python hscm_full.py --shift-demo
      python hscm_full.py --bytecode-demo
"""

import sys, os, re, types, marshal, hashlib, json, argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from matrix import (L0_PRIMITIVES, L4_TRAJECTORY,
                    SCAFFOLDING_SUBPOS, LOW_LOAD_PRIMITIVES,
                    VALID_L0_L1_EDGES, is_load_bearing)
from engine import Coordinate, TraversalEngine, Compressor


# =============================================================================
# 1. CONTEXT TABLE
#    The active set of premise refinements.
#    Each entry is an S-expression stored as a Python list (homoiconic).
#    Push = fine-tune. Pop = rollback. List = full history.
# =============================================================================

@dataclass
class ContextShift:
    """One fine-tuning adjustment — stored as S-expression data."""
    timestamp:    str
    premise:      str
    operation:    str      # "DEMOTE" | "PROMOTE" | "SWAP_TRAIT"
    target:       str      # primitive or subpos being adjusted
    from_value:   Any
    to_value:     Any
    triggered_by: str      # what input caused this shift
    residual_before: List[str]
    residual_after:  List[str]
    committed:    bool = False

    def to_sexpr(self) -> list:
        """This object AS a LISP S-expression — data and instruction unified."""
        return [
            'CONTEXT-SHIFT',
            self.premise,
            self.operation,
            self.target,
            self.from_value,
            self.to_value,
        ]

    def __repr__(self):
        return (f"[{self.operation}] {self.premise}/{self.target}: "
                f"{self.from_value} → {self.to_value} "
                f"({'committed' if self.committed else 'provisional'})")


class ContextTable:
    """
    The active context — a live LISP-style list of premise refinements.
    Shifts are prepended (LISP push semantics) so rollback is always pop.
    The full history is the fine-tuning record.
    """

    SHIFT_LOG = Path("logs/context_shifts.jsonl")

    def __init__(self):
        # Per-premise scaffolding overrides
        # {premise: set of primitives demoted to scaffolding}
        self._demoted:  Dict[str, set] = {}
        # {premise: set of primitives promoted to load-bearing}
        self._promoted: Dict[str, set] = {}
        # Full shift history as S-expressions
        self.history:   List[ContextShift] = []
        self.SHIFT_LOG.parent.mkdir(exist_ok=True)

    def push_shift(self, shift: ContextShift):
        """Add a shift — LISP push semantics, prepend to history."""
        premise = shift.premise

        if shift.operation == "DEMOTE":
            if premise not in self._demoted:
                self._demoted[premise] = set()
            self._demoted[premise].add(shift.target)

        elif shift.operation == "PROMOTE":
            if premise not in self._promoted:
                self._promoted[premise] = set()
            self._promoted[premise].add(shift.target)

        self.history.insert(0, shift)  # prepend — LISP list order
        self._log(shift)

    def pop_shift(self) -> Optional[ContextShift]:
        """Roll back the most recent shift."""
        if not self.history:
            return None
        shift = self.history.pop(0)
        premise = shift.premise

        if shift.operation == "DEMOTE":
            self._demoted.get(premise, set()).discard(shift.target)
        elif shift.operation == "PROMOTE":
            self._promoted.get(premise, set()).discard(shift.target)

        print(f"  ROLLBACK: {shift}")
        return shift

    def is_demoted(self, primitive: str, premise: str) -> bool:
        return primitive in self._demoted.get(premise, set())

    def is_promoted(self, primitive: str, premise: str) -> bool:
        return primitive in self._promoted.get(premise, set())

    def as_sexprs(self) -> List[list]:
        """The entire fine-tuning history as S-expressions — portable, readable."""
        return [s.to_sexpr() for s in self.history]

    def _log(self, shift: ContextShift):
        with open(self.SHIFT_LOG, 'a') as f:
            f.write(json.dumps(asdict(shift)) + '\n')

    def save(self, path: str = "logs/context_table.json"):
        with open(path, 'w') as f:
            json.dump(self.as_sexprs(), f, indent=2)
        print(f"  Context table saved → {path}")

    def report(self):
        print(f"\n  CONTEXT TABLE ({len(self.history)} shifts)")
        if not self.history:
            print("  (no shifts — using base context)")
            return
        for s in self.history:
            print(f"  {s}")
        print(f"\n  As S-expressions:")
        for sx in self.as_sexprs():
            print(f"    {sx}")


# =============================================================================
# 2. CONTEXT-AWARE COLLAPSE
#    The collapse function now checks the context table before deciding
#    if a coordinate is load-bearing or scaffolding.
# =============================================================================

@dataclass
class CollapseResult:
    original:         List[Coordinate]
    residual:         List[Coordinate]
    scaffolding:      List[Coordinate]
    intent_primitive: Optional[str]
    intent_trajectory: Dict[str, str]
    premise:          str
    shifts_applied:   List[str]   # which context shifts affected this collapse

    def __repr__(self):
        return (
            f"CollapseResult(\n"
            f"  words={[c.surface_form for c in self.original]}\n"
            f"  residual={[c.surface_form for c in self.residual]}\n"
            f"  intent={self.intent_primitive} / {self.intent_trajectory}\n"
            f"  shifts_applied={self.shifts_applied}\n"
            f")"
        )


def collapse_with_context(sequence: List[Coordinate],
                          premise: str,
                          ctx: ContextTable) -> CollapseResult:
    """
    Context-aware collapse.
    Checks context table for demotions/promotions before the base
    is_load_bearing test. This is the fine-tunable collapse function.
    """
    active_primitives: set = set()
    active_axes: dict = {}
    residual:    List[Coordinate] = []
    scaffolding: List[Coordinate] = []
    shifts_applied: List[str] = []

    for coord in sequence:
        prim = coord.l0_primitive

        # Check context table overrides FIRST
        if ctx.is_demoted(prim, premise):
            scaffolding.append(coord)
            shifts_applied.append(f"DEMOTED:{prim}")
            continue

        if ctx.is_promoted(prim, premise):
            residual.append(coord)
            active_primitives.add(prim)
            shifts_applied.append(f"PROMOTED:{prim}")
            continue

        # Base load-bearing test
        lb = is_load_bearing(
            prim, coord.l1_subpos, coord.l4_trajectory,
            active_primitives, active_axes
        )

        if lb:
            residual.append(coord)
            active_primitives.add(prim)
            for traj in coord.l4_trajectory:
                if traj in L4_TRAJECTORY:
                    axis = L4_TRAJECTORY[traj]["axis"]
                    pole = L4_TRAJECTORY[traj]["pole"]
                    active_axes[axis] = pole
        else:
            scaffolding.append(coord)

    # Determine intent primitive
    intent_primitive = None
    priority = ["ACTION", "MENTAL", "SPEECH", "EVALUATOR"]
    for coord in residual:
        cat = L0_PRIMITIVES.get(coord.l0_primitive, {}).get("category", "")
        if cat in priority:
            intent_primitive = coord.l0_primitive
            break
    if intent_primitive is None and residual:
        intent_primitive = residual[0].l0_primitive

    return CollapseResult(
        original=sequence,
        residual=residual,
        scaffolding=scaffolding,
        intent_primitive=intent_primitive,
        intent_trajectory=active_axes,
        premise=premise,
        shifts_applied=shifts_applied,
    )


# =============================================================================
# 3. CONTEXT SHIFT LOOP
#    Run collapse → check result → shift if wrong → re-run → commit or rollback
# =============================================================================

class ContextShiftLoop:
    """
    Fine-tuning loop.
    Not training — targeted contextual refinement.
    The shift log IS the fine-tuning record.
    """

    def __init__(self, ctx: ContextTable):
        self.ctx = ctx

    def run(self,
            sequence: List[Coordinate],
            expected_intent: str,
            premise: str = "ANALYTICAL",
            max_shifts: int = 5) -> CollapseResult:
        """
        Run collapse. If residual intent doesn't match expected,
        shift context and retry. Up to max_shifts attempts.
        """
        print(f"\n  SHIFT LOOP: expecting intent={expected_intent}, "
              f"premise={premise}")
        print(f"  Input: {[c.surface_form for c in sequence]}")

        for attempt in range(max_shifts + 1):
            result = collapse_with_context(sequence, premise, self.ctx)

            print(f"  Attempt {attempt+1}: residual={[c.surface_form for c in result.residual]}, "
                  f"intent={result.intent_primitive}")

            if result.intent_primitive == expected_intent:
                print(f"  ✓ Correct intent reached")
                # Commit all shifts made during this loop
                for s in self.ctx.history:
                    if not s.committed:
                        s.committed = True
                return result

            if attempt == max_shifts:
                print(f"  ✗ Max shifts reached. Rolling back.")
                # Roll back uncommitted shifts
                while self.ctx.history and not self.ctx.history[0].committed:
                    self.ctx.pop_shift()
                return result

            # Identify what to shift
            # The wrong load-bearing coordinate is the first residual item
            # that isn't the expected intent
            wrong_coord = None
            for coord in result.residual:
                if coord.l0_primitive != expected_intent:
                    wrong_coord = coord
                    break

            if wrong_coord is None:
                break

            shift = ContextShift(
                timestamp=datetime.now().isoformat(),
                premise=premise,
                operation="DEMOTE",
                target=wrong_coord.l0_primitive,
                from_value="LOAD_BEARING",
                to_value="SCAFFOLDING",
                triggered_by=f"wrong_intent:{result.intent_primitive}",
                residual_before=[c.surface_form for c in result.residual],
                residual_after=[],  # filled after next collapse
            )

            print(f"  → Shifting: DEMOTE {wrong_coord.l0_primitive} "
                  f"under {premise}")
            self.ctx.push_shift(shift)

        return result


# =============================================================================
# 4. RAG — Retrieval-Augmented Generation over the manifold
#    Query → retrieve relevant coordinates → augment → generate
#
#    "Embedding" = the coordinate tuple itself (already semantic)
#    No vector database needed — the manifold IS the embedding space
#    Retrieval = constraint-based traversal, not cosine similarity
# =============================================================================

@dataclass
class RAGDocument:
    """A chunk of knowledge stored with its manifold coordinate."""
    doc_id:    str
    text:      str
    primitive: str
    subpos:    str
    trajectory: List[str]
    source:    str = "manual"

    def coordinate_key(self) -> tuple:
        return (self.primitive, self.subpos, tuple(sorted(self.trajectory)))


class RAGStore:
    """
    The RAG store. Documents are indexed by manifold coordinate.
    Retrieval is coordinate proximity, not vector similarity.
    This means retrieval is deterministic and explainable.
    """

    def __init__(self):
        self._store: Dict[tuple, List[RAGDocument]] = {}
        self._all:   List[RAGDocument] = []
        self.INDEX_PATH = Path("index/rag_store.json")

    def add(self, doc: RAGDocument):
        key = doc.coordinate_key()
        if key not in self._store:
            self._store[key] = []
        self._store[key].append(doc)
        self._all.append(doc)

    def retrieve(self,
                 primitive: str,
                 trajectory: List[str],
                 top_k: int = 3) -> List[RAGDocument]:
        """
        Retrieve documents by coordinate proximity.
        Normalizes trajectory labels (VEL_FAST → FAST, PER_STATIC → STATIC)
        for matching against seed documents.
        """
        # Normalize full key → short label
        def normalize(t):
            return t.split('_', 1)[-1] if '_' in t else t

        norm_traj = [normalize(t) for t in trajectory]
        results = []

        for doc in self._all:
            if doc.primitive != primitive:
                continue
            norm_doc = [normalize(t) for t in doc.trajectory]
            match_count = sum(1 for t in norm_traj if t in norm_doc)
            # Also match on primitive alone if no trajectory given
            if not trajectory:
                match_count = 1
            if match_count > 0:
                results.append((match_count, doc))

        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:top_k]]

    def retrieve_from_collapse(self,
                               collapse: CollapseResult,
                               top_k: int = 3) -> List[RAGDocument]:
        """Retrieve using the intent from a context collapse."""
        traj = list(collapse.intent_trajectory.values())
        return self.retrieve(collapse.intent_primitive or "", traj, top_k)

    def save(self):
        data = [asdict(d) for d in self._all]
        self.INDEX_PATH.parent.mkdir(exist_ok=True)
        with open(self.INDEX_PATH, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not self.INDEX_PATH.exists():
            return
        with open(self.INDEX_PATH) as f:
            for item in json.load(f):
                self.add(RAGDocument(**item))
        print(f"  RAG store loaded: {len(self._all)} documents")

    def report(self):
        print(f"\n  RAG STORE: {len(self._all)} documents across "
              f"{len(self._store)} coordinate regions")


# =============================================================================
# 5. RAG QUERY ENGINE
#    Collapse input → retrieve → augment response
# =============================================================================

class RAGEngine:

    def __init__(self,
                 compressor: Compressor,
                 engine:     TraversalEngine,
                 rag_store:  RAGStore,
                 ctx:        ContextTable):
        self.compressor = compressor
        self.engine     = engine
        self.rag_store  = rag_store
        self.ctx        = ctx
        self.shift_loop = ContextShiftLoop(ctx)

    def query(self,
              words: List[str],
              word_coords: List[Coordinate],
              premise: str = "ANALYTICAL",
              expected_intent: str = None) -> dict:
        """
        Full RAG pipeline:
          1. Collapse input to intent
          2. Optionally run shift loop to correct intent
          3. Retrieve relevant documents from RAG store
          4. Return augmented result
        """
        # Step 1: Collapse
        if expected_intent:
            collapse = self.shift_loop.run(
                word_coords, expected_intent, premise
            )
        else:
            collapse = collapse_with_context(word_coords, premise, self.ctx)

        # Step 2: Retrieve
        retrieved = self.rag_store.retrieve_from_collapse(collapse, top_k=3)

        # Step 3: Augment
        return {
            "query":     words,
            "intent":    collapse.intent_primitive,
            "trajectory": collapse.intent_trajectory,
            "residual":  [c.surface_form for c in collapse.residual],
            "retrieved": [
                {"text": d.text, "source": d.source,
                 "primitive": d.primitive, "trajectory": d.trajectory}
                for d in retrieved
            ],
            "shifts_applied": collapse.shifts_applied,
            "premise":   premise,
        }


# =============================================================================
# 6. BYTECODE EMBEDDING
#    Precompile traversal functions → store as bytes in LISP lists
#    Python points at the list item and fires — no recompilation
# =============================================================================

class BytecodeEmbedding:
    """
    Compile Python traversal functions once.
    Store bytecode in LISP-style lists.
    Execute at near-C speed — no interpreter overhead on hot path.
    """

    def __init__(self):
        # The LISP list: each entry is [FUNCTION-NAME, bytecode_bytes]
        self._lisp_bytecode_list: List[list] = []
        self._fn_cache: Dict[str, types.FunctionType] = {}
        self.CACHE_PATH = Path("index/bytecode_cache.bin")

    def compile_and_embed(self, name: str, source: str, globs: dict):
        """
        Compile Python source → bytecode → store in LISP list.
        This is the 'embedding' — the function's identity is its bytecode.
        """
        code    = compile(source, f"<hscm:{name}>", "eval")
        bc      = marshal.dumps(code)

        # The LISP S-expression for this function:
        # (BYTECODE-EMBED "name" <bytes> <hash>)
        sexpr = [
            'BYTECODE-EMBED',
            name,
            bc,
            hashlib.md5(bc).hexdigest()[:8]
        ]
        self._lisp_bytecode_list.append(sexpr)

        # Build executable function from bytecode
        def fn(_code=code, _globs=globs, **kwargs):
            local_globs = {**_globs, **kwargs}
            return eval(_code, local_globs)

        self._fn_cache[name] = fn
        return fn

    def get(self, name: str) -> Optional[types.FunctionType]:
        """Get precompiled function — O(1) lookup, no recompilation."""
        return self._fn_cache.get(name)

    def execute(self, name: str, **kwargs):
        """Fire precompiled function directly. This is the fast path."""
        fn = self._fn_cache.get(name)
        if fn is None:
            raise KeyError(f"No bytecode embedding for: {name}")
        return fn(**kwargs)

    def lisp_list(self) -> List[list]:
        """The full LISP structure — all embeddings as S-expressions."""
        return self._lisp_bytecode_list

    def report(self):
        print(f"\n  BYTECODE EMBEDDINGS: {len(self._lisp_bytecode_list)} functions")
        for sx in self._lisp_bytecode_list:
            print(f"    ({sx[0]} {sx[1]} <{len(sx[2])} bytes> hash={sx[3]})")


# =============================================================================
# SEED DATA — physics + linguistics knowledge base for RAG
# =============================================================================

SEED_DOCUMENTS = [
    # Physics
    RAGDocument("phy_001",
        "An object at rest stays at rest. An object in motion stays in motion "
        "unless acted upon by an external force. This is Newton's First Law.",
        "MOVE", "VERB_STATE", ["STATIC", "DYNAMIC"], "Newton"),
    RAGDocument("phy_002",
        "Force equals mass times acceleration. F=ma. "
        "Applying force to a mass produces acceleration proportional to the force.",
        "DO", "VERB_CAUSATIVE", ["DYNAMIC", "LARGE", "OUTWARD"], "Newton"),
    RAGDocument("phy_003",
        "For every action there is an equal and opposite reaction. "
        "Forces always occur in pairs.",
        "HAPPEN", "VERB_PROCESS", ["DYNAMIC", "INWARD"], "Newton"),
    RAGDocument("phy_004",
        "Velocity is the rate of change of position. "
        "Fast velocity means large displacement per unit time.",
        "MOVE", "VERB_MOTION", ["FAST", "DYNAMIC", "OUTWARD"], "kinematics"),
    RAGDocument("phy_005",
        "A constraint limits the degrees of freedom of a system. "
        "Limitations on motion reduce the possible states a system can occupy.",
        "MOVE", "VERB_STATE", ["STATIC", "INWARD"], "mechanics"),

    # Linguistics
    RAGDocument("ling_001",
        "Semantic primitives are irreducible conceptual atoms present in all "
        "human languages. Wierzbicka identified 65 such universals.",
        "KNOW", "VERB_MENTAL", ["STATIC", "INWARD"], "NSM"),
    RAGDocument("ling_002",
        "Motion verbs encode direction, manner, and speed. "
        "Sprint encodes high speed. Crawl encodes low speed with effort.",
        "MOVE", "VERB_MOTION", ["DYNAMIC", "OUTWARD"], "lexical_semantics"),
    RAGDocument("ling_003",
        "Mental predicates — think, know, want, feel — describe internal states. "
        "They are universally inward-directed in trajectory.",
        "THINK", "VERB_MENTAL", ["INWARD", "DYNAMIC"], "NSM"),
    RAGDocument("ling_004",
        "Context collapse reduces an utterance to its load-bearing coordinates. "
        "The residual after collapse IS the intent of the utterance.",
        "SAY", "VERB_SPEECH", ["OUTWARD", "DYNAMIC"], "HSCM"),
    RAGDocument("ling_005",
        "A collision in the manifold occurs when two coordinate paths arrive at "
        "the same (L0, L1) with opposing L4 trajectory poles. "
        "Under empirical premise this is contradiction. Under creative premise it is metaphor.",
        "KNOW", "VERB_MENTAL", ["STATIC", "INWARD"], "HSCM"),

    # Mathematics
    RAGDocument("math_001",
        "A right angle is exactly 90 degrees. It is the angle between two "
        "perpendicular lines. The foundation of Euclidean geometry.",
        "BE", "VERB_STATE", ["STATIC", "LARGE"], "Euclid"),
    RAGDocument("math_002",
        "The Pythagorean theorem: in a right triangle, the square of the "
        "hypotenuse equals the sum of squares of the other two sides. a²+b²=c²",
        "BE", "VERB_STATE", ["STATIC"], "Euclid"),
    RAGDocument("math_003",
        "A proof proceeds from axioms through valid logical steps to a conclusion. "
        "Each step must follow necessarily from the prior.",
        "KNOW", "VERB_MENTAL", ["STATIC", "INWARD"], "logic"),
]


# =============================================================================
# MAIN — wire everything together
# =============================================================================

def build_system():
    """Build the full system and return components."""
    compressor = Compressor()
    engine     = TraversalEngine()
    ctx        = ContextTable()
    rag_store  = RAGStore()
    bytecode   = BytecodeEmbedding()

    # Seed RAG store
    for doc in SEED_DOCUMENTS:
        rag_store.add(doc)

    # Seed manifold with vocabulary
    vocab = [
        ("sprint",  "MOVE",  "VERB_MOTION", "ROOT_GERMANIC", ["VEL_FAST","PER_DYNAMIC","DIR_OUT"]),
        ("crawl",   "MOVE",  "VERB_MOTION", "ROOT_GERMANIC", ["VEL_SLOW","PER_DYNAMIC","DIR_OUT"]),
        ("run",     "MOVE",  "VERB_MOTION", "ROOT_GERMANIC", ["VEL_FAST","PER_DYNAMIC","DIR_OUT"]),
        ("amble",   "MOVE",  "VERB_MOTION", "ROOT_FRENCH",   ["VEL_SLOW","PER_DYNAMIC","DIR_OUT"]),
        ("flee",    "MOVE",  "VERB_MOTION", "ROOT_GERMANIC", ["VEL_FAST","PER_DYNAMIC","DIR_OUT","AFF_NEG"]),
        ("think",   "THINK", "VERB_MENTAL", "ROOT_GERMANIC", ["DIR_IN","PER_DYNAMIC"]),
        ("ponder",  "THINK", "VERB_MENTAL", "ROOT_LATIN",    ["DIR_IN","PER_STATIC"]),
        ("want",    "WANT",  "VERB_MENTAL", "ROOT_GERMANIC", ["DIR_IN","PER_STATIC"]),
        ("crave",   "WANT",  "VERB_MENTAL", "ROOT_GERMANIC", ["DIR_IN","PER_DYNAMIC","MAG_LARGE"]),
        ("know",    "KNOW",  "VERB_MENTAL", "ROOT_GERMANIC", ["DIR_IN","PER_STATIC"]),
        ("say",     "SAY",   "VERB_SPEECH", "ROOT_GERMANIC", ["DIR_OUT","PER_DYNAMIC"]),
        ("shout",   "SAY",   "VERB_SPEECH", "ROOT_GERMANIC", ["DIR_OUT","PER_DYNAMIC","MAG_LARGE"]),
        ("whisper", "SAY",   "VERB_SPEECH", "ROOT_GERMANIC", ["DIR_OUT","PER_DYNAMIC","MAG_SMALL"]),
    ]

    coords = {}
    for word, prim, subpos, geo, traj in vocab:
        r = compressor.compress(word, prim, subpos, geo, traj)
        if r["validation"]["valid"]:
            engine.register(r["coordinate"])
            coords[word] = r["coordinate"]

    # Precompile bytecode embeddings for hot-path traversal functions
    manifold_globs = {"registry": engine.registry, "L4_TRAJECTORY": L4_TRAJECTORY}

    bytecode.compile_and_embed(
        "count_valid",
        "len(registry)",
        manifold_globs
    )
    bytecode.compile_and_embed(
        "find_by_primitive",
        "[c for c in registry.values() if c.l0_primitive == primitive]",
        manifold_globs
    )
    bytecode.compile_and_embed(
        "find_fast_motion",
        "[c for c in registry.values() "
        " if c.l0_primitive == 'MOVE' and 'VEL_FAST' in c.l4_trajectory]",
        manifold_globs
    )

    rag_engine = RAGEngine(compressor, engine, rag_store, ctx)

    return compressor, engine, ctx, rag_store, bytecode, rag_engine, coords


def demo_rag(query_text: str = None):
    """RAG demo — query the manifold-backed knowledge store."""
    _, engine, ctx, rag_store, _, _, coords = build_system()

    print("\n" + "="*60)
    print("  RAG — Retrieval over the Manifold")
    print("="*60)

    queries = [
        ("slow deliberate motion", "MOVE", ["VEL_SLOW", "PER_DYNAMIC"]),
        ("internal mental desire", "WANT", ["DIR_IN", "PER_STATIC"]),
        ("physical force and change", "DO",  ["PER_DYNAMIC", "MAG_LARGE"]),
    ]

    if query_text:
        queries = [(query_text, "MOVE", [])]

    for text, primitive, traj in queries:
        print(f"\n  Query: '{text}'")
        docs = rag_store.retrieve(primitive, traj, top_k=2)
        print(f"  Retrieved {len(docs)} documents:")
        for doc in docs:
            print(f"    [{doc.source}] {doc.text[:80]}...")


def demo_shift_loop():
    """Context shift fine-tuning demo."""
    _, _, ctx, _, _, _, coords = build_system()

    print("\n" + "="*60)
    print("  CONTEXT SHIFT LOOP — Fine-tuning without training")
    print("="*60)

    loop = ContextShiftLoop(ctx)

    # Input: want + think + sprint
    # Expected intent: WANT (desire driving motion)
    # Without shift: THINK fires first as load-bearing
    sequence = [coords["want"], coords["think"], coords["sprint"]]

    result = loop.run(
        sequence,
        expected_intent="WANT",
        premise="ANALYTICAL",
        max_shifts=3
    )

    print(f"\n  Final residual: {[c.surface_form for c in result.residual]}")
    print(f"  Intent: {result.intent_primitive}")
    ctx.report()

    print(f"\n  ROLLBACK TEST:")
    ctx.pop_shift()
    ctx.report()


def demo_bytecode():
    """Bytecode embedding demo — precompiled functions in LISP lists."""
    _, _, _, _, bytecode, _, _ = build_system()

    print("\n" + "="*60)
    print("  BYTECODE EMBEDDINGS — LISP list as execution layer")
    print("="*60)

    bytecode.report()

    print(f"\n  Executing precompiled 'count_valid':")
    count = bytecode.execute("count_valid")
    print(f"    Result: {count} registered coordinates")

    print(f"\n  Executing precompiled 'find_fast_motion':")
    fast = bytecode.execute("find_fast_motion")
    print(f"    Fast motion words: {[c.surface_form for c in fast]}")

    print(f"\n  The LISP list (each entry is bytecode + hash):")
    for sx in bytecode.lisp_list():
        print(f"    [{sx[1]}] {len(sx[2])} bytes, hash={sx[3]}")

    print(f"\n  Python points at this list and fires.")
    print(f"  No recompilation. No interpretation overhead.")
    print(f"  The list IS the embedding. The bytecode IS the weight.")


def demo_full():
    """Full integrated pipeline."""
    compressor, engine, ctx, rag_store, bytecode, rag_engine, coords = build_system()

    print("\n" + "="*60)
    print("  HSCM FULL SYSTEM — Weekend Experiment")
    print("="*60)
    print(f"  Manifold: {len(engine.registry)} coordinates")

    demo_bytecode()
    demo_shift_loop()
    demo_rag()

    print("\n" + "="*60)
    print("  SAVING STATE")
    print("="*60)
    ctx.save()
    rag_store.save()
    print("  Done. Files written to logs/ and index/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSCM Full System")
    parser.add_argument("--rag",        type=str, help="RAG query")
    parser.add_argument("--shift-demo", action="store_true")
    parser.add_argument("--bytecode-demo", action="store_true")
    args = parser.parse_args()

    if args.rag:
        demo_rag(args.rag)
    elif args.shift_demo:
        demo_shift_loop()
    elif args.bytecode_demo:
        demo_bytecode()
    else:
        demo_full()