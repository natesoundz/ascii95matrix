"""
HSCM Demonstration
Shows compression, traversal, and coherence checking with real examples.
All dead ends are reported as findings, not suppressed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from engine import Compressor, TraversalEngine, PhoneticParser
from matrix import PREMISE_CONTEXTS


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_phonetic_parser():
    separator("PHONETIC PARSER — L2 Sequence Extraction")
    parser = PhoneticParser()

    # The think/thank distinction that broke the previous model
    test_words = [
        ("think", "Should produce T-TH_V-V_HIGH_FRONT-N-K"),
        ("thank", "Should produce T-TH_V-V_LOW-N-K"),
        ("sprint", ""),
        ("crawl", ""),
        ("want", ""),
        ("know", ""),
        ("feel", ""),
        ("move", ""),
        ("through", ""),
        ("threw", "Phonetically similar to 'through' — algebra must distinguish"),
    ]

    for word, note in test_words:
        seq = parser.parse(word)
        print(f"  {word:<12} → {'-'.join(seq):<30}  {note}")


def demo_compression():
    separator("COMPRESSION — Word → Coordinate")
    compressor = Compressor()

    examples = [
        {
            "word": "sprint",
            "primitive": "MOVE",
            "subpos": "VERB_MOTION",
            "geometric": "ROOT_GERMANIC",
            "trajectory": ["VEL_FAST", "PER_DYNAMIC", "DIR_OUT"],
            "premise": "ANALYTICAL",
        },
        {
            "word": "crawl",
            "primitive": "MOVE",
            "subpos": "VERB_MOTION",
            "geometric": "ROOT_GERMANIC",
            "trajectory": ["VEL_SLOW", "PER_DYNAMIC", "DIR_OUT"],
            "premise": "ANALYTICAL",
        },
        {
            "word": "want",
            "primitive": "WANT",
            "subpos": "VERB_MENTAL",
            "geometric": "ROOT_GERMANIC",
            "trajectory": ["DIR_IN", "PER_STATIC"],
            "premise": "ANALYTICAL",
        },
        # Deliberate dead end — invalid L0/L1 edge
        {
            "word": "want",
            "primitive": "WANT",
            "subpos": "VERB_MOTION",  # WRONG — WANT cannot be a motion verb
            "geometric": "ROOT_GERMANIC",
            "trajectory": ["DIR_IN"],
            "premise": "ANALYTICAL",
        },
    ]

    for ex in examples:
        result = compressor.compress(
            word=ex["word"],
            primitive=ex["primitive"],
            subpos=ex["subpos"],
            geometric=ex["geometric"],
            trajectory=ex["trajectory"],
            premise=ex["premise"]
        )
        print(f"\n  Word: '{ex['word']}'")
        print(f"  Coordinate: {result['coordinate']}")
        print(f"  Phonetic: {'-'.join(result['phonetic_sequence'])}")
        print(f"  Valid: {result['validation']['valid']}")
        if result['validation']['dead_ends']:
            for de in result['validation']['dead_ends']:
                print(f"  ⚠  {de['finding']}")
        if result['tuple']:
            print(f"  Stored tuple: {result['tuple']}")


def demo_traversal():
    separator("TRAVERSAL — Constraint Intersection Query")

    compressor = Compressor()
    engine = TraversalEngine()

    # Populate the space with a small vocabulary
    vocabulary = [
        ("sprint",  "MOVE", "VERB_MOTION", "ROOT_GERMANIC", ["VEL_FAST", "PER_DYNAMIC", "DIR_OUT"]),
        ("crawl",   "MOVE", "VERB_MOTION", "ROOT_GERMANIC", ["VEL_SLOW", "PER_DYNAMIC", "DIR_OUT"]),
        ("run",     "MOVE", "VERB_MOTION", "ROOT_GERMANIC", ["VEL_FAST", "PER_DYNAMIC", "DIR_OUT"]),
        ("amble",   "MOVE", "VERB_MOTION", "ROOT_FRENCH",   ["VEL_SLOW", "PER_DYNAMIC", "DIR_OUT"]),
        ("think",   "THINK","VERB_MENTAL", "ROOT_GERMANIC", ["DIR_IN", "PER_DYNAMIC"]),
        ("ponder",  "THINK","VERB_MENTAL", "ROOT_LATIN",    ["DIR_IN", "PER_STATIC"]),
        ("want",    "WANT", "VERB_MENTAL", "ROOT_GERMANIC", ["DIR_IN", "PER_STATIC"]),
        ("crave",   "WANT", "VERB_MENTAL", "ROOT_GERMANIC", ["DIR_IN", "PER_DYNAMIC", "MAG_LARGE"]),
    ]

    for word, prim, subpos, geo, traj in vocabulary:
        result = compressor.compress(word, prim, subpos, geo, traj)
        if result["validation"]["valid"]:
            engine.register(result["coordinate"])

    print(f"\n  Registered {len(engine.registry)} coordinates.\n")

    # Query 1: All fast motion verbs
    print("  QUERY: All fast outward motion verbs")
    q1 = engine.query({"l0": "MOVE", "l4": ["VEL_FAST", "DIR_OUT"]})
    print(f"  Finding: {q1['finding']}")
    for m in q1["matches"]:
        print(f"    → {m.surface_form} {repr(m)}")

    # Query 2: All inward mental states (slow — desire/want region)
    print("\n  QUERY: Static inward mental states")
    q2 = engine.query({"l0": "WANT", "l4": ["DIR_IN", "PER_STATIC"]})
    print(f"  Finding: {q2['finding']}")
    for m in q2["matches"]:
        print(f"    → {m.surface_form} {repr(m)}")

    # Query 3: A dead end — invalid combination
    print("\n  QUERY: Fast static motion (contradictory — should be empty)")
    q3 = engine.query({"l0": "MOVE", "l4": ["VEL_FAST", "PER_STATIC"]})
    print(f"  Finding: {q3['finding']}")
    print(f"  (Empty result is the correct output — the dead end IS the finding)")


def demo_coherence_check():
    separator("COHERENCE CHECK — Premise-Scoped Inconsistency Detection")

    compressor = Compressor()

    # Build a sentence: "The static fire spread rapidly"
    # STATIC and DYNAMIC on the same referent — structural contradiction
    # under EMPIRICAL premise

    fire_static = compressor.compress(
        "fire", "HAPPEN", "NOUN_EVENT", "ROOT_GERMANIC",
        ["PER_STATIC"],  # declared static
        premise="EMPIRICAL"
    )["coordinate"]

    spread_rapid = compressor.compress(
        "spread", "MOVE", "VERB_MOTION", "ROOT_GERMANIC",
        ["VEL_FAST", "PER_DYNAMIC"],  # declared dynamic and fast
        premise="EMPIRICAL"
    )["coordinate"]

    engine = TraversalEngine()

    print("\n  Sentence: 'The static fire spread rapidly'")
    print("  Premise: EMPIRICAL\n")

    result = engine.coherence_check([fire_static, spread_rapid], "EMPIRICAL")
    print(f"  Coherent: {result['coherent']}")
    if result["issues"]:
        for issue in result["issues"]:
            print(f"  ⚠  Position {issue['position']} ('{issue['word']}'): {issue['issue']}")

    # Now check under CREATIVE premise — same sentence should pass
    print("\n  Same sentence under CREATIVE premise:")
    fire_static.premise = "CREATIVE"
    spread_rapid.premise = "CREATIVE"
    result2 = engine.coherence_check([fire_static, spread_rapid], "CREATIVE")
    print(f"  Coherent: {result2['coherent']}")
    print(f"  Premise: {result2['premise_description']}")
    print("  (No contradiction flagged — creative context suspends physics constraints)")


def demo_moon_cheese():
    separator("PREMISE FIELD TEST — 'The Moon Is Made of Cheese'")

    compressor = Compressor()

    moon = compressor.compress(
        "moon", "THING", "NOUN_INANIMATE", "ROOT_GERMANIC",
        ["MAG_LARGE", "PER_STATIC"],
        premise="EMPIRICAL"
    )["coordinate"]

    cheese = compressor.compress(
        "cheese", "THING", "NOUN_CONCRETE", "ROOT_GERMANIC",
        ["MAG_SMALL", "PER_STATIC"],
        premise="EMPIRICAL"
    )["coordinate"]

    engine = TraversalEngine()

    print("\n  Premise: EMPIRICAL")
    print("  'The moon is made of cheese'")
    result = engine.coherence_check([moon, cheese], "EMPIRICAL")
    print(f"  Internally coherent: {result['coherent']}")
    print(f"  Note: Internal structural coherence passes.")
    print(f"  The EMPIRICAL premise catches self-contradiction, not empirical falsehood.")
    print(f"  'Moon=cheese' has no internal axis contradiction — it is wrong,")
    print(f"  but wrong-against-external-reality requires an anchored fact layer.")
    print(f"  That is a second system built on top. This is the honest boundary.")

    print("\n  Premise: CREATIVE")
    result2 = engine.coherence_check([moon, cheese], "CREATIVE")
    print(f"  Coherent: {result2['coherent']}")
    print(f"  Premise: {result2['premise_description']}")
    print(f"  (Valid as metaphor or literary device)")


if __name__ == "__main__":
    print("\nHSCM — Hierarchical Semantic Coordinate Matrix")
    print("Operating premise: Inspired Skeptic")
    print("Dead ends are findings. Only valid paths produce output.\n")

    demo_phonetic_parser()
    demo_compression()
    demo_traversal()
    demo_coherence_check()
    demo_moon_cheese()

    print("\n" + "="*60)
    print("  HONEST STUBS — What this system still needs:")
    print("="*60)
    print("""
  1. CMU Pronouncing Dictionary integration
     → Replaces character-level phonetic parsing with 
       phoneme-accurate L2 sequences
     → Required for 'through'/'threw' disambiguation at scale

  2. Wiktionary/WordNet L0 annotation layer
     → Automates primitive assignment for large vocabularies
     → Currently requires human or upstream classifier

  3. L3 etymological lookup
     → Requires etymology database (Etymonline API or Wiktionary)
     → Currently declared manually

  4. Empirical falsehood layer (System 2)
     → Anchored fact nodes at L0 primitives
     → Requires external knowledge base
     → Built on top of this system, not inside it
  """)
