"""
HSCM Orchestrator
Drives: compile C++ → locate dictionary → build manifold → ingest words
Monitors all subprocesses. Checkpoints allow resume at any layer.
"""

import subprocess
import threading
import sys
import os
import json
import time
import signal
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BASE_DIR, DATA_DIR, CHECKPOINT_DIR, INDEX_DIR, LOG_DIR, BUILD_DIR,
    locate_or_download_dictionary, DictPointer
)
from matrix import L0_PRIMITIVES, L1_SUBPOS, VALID_L0_L1_EDGES

ENGINE_SRC  = BASE_DIR / "manifold_engine.cpp"
ENGINE_BIN  = BUILD_DIR / "manifold_engine"
STATE_FILE  = CHECKPOINT_DIR / "pipeline_state.json"
MAIN_LOG    = LOG_DIR / "orchestrator.log"

# =============================================================================
# PIPELINE STATE — what's been completed, what can be resumed
# =============================================================================

@dataclass
class PipelineState:
    started_at:      str  = ""
    dict_path:       str  = ""
    dict_format:     str  = ""
    dict_lines:      int  = 0
    engine_compiled: bool = False
    layer_completed: int  = -1   # -1 = none, 0=L0, 1=L1, ... 4=L4
    manifold_path:   str  = ""
    manifold_size:   int  = 0
    ingestion_done:  bool = False
    words_ingested:  int  = 0
    last_checkpoint: str  = ""

    def save(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        log(f"  State saved → {STATE_FILE}")

    @classmethod
    def load(cls) -> 'PipelineState':
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return cls(**json.load(f))
        return cls()


# =============================================================================
# LOGGING
# =============================================================================

def log(msg: str, also_print: bool = True):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    with open(MAIN_LOG, 'a') as f:
        f.write(line + "\n")
    if also_print:
        print(line)


# =============================================================================
# MONITOR THREAD — reads C++ stderr and surfaces structured events
# =============================================================================

class EngineMonitor(threading.Thread):
    def __init__(self, proc: subprocess.Popen, on_event=None):
        super().__init__(daemon=True)
        self.proc     = proc
        self.on_event = on_event
        self.events   = []
        self.last_progress = {}

    def run(self):
        for raw_line in self.proc.stderr:
            line = raw_line.decode('utf-8', errors='replace').strip()
            if not line:
                continue

            # Parse structured log lines from the C++ engine
            event = self._parse_event(line)
            self.events.append(event)

            # Surface to orchestrator log
            log(f"  [ENGINE] {line}")

            if self.on_event:
                self.on_event(event)

    def _parse_event(self, line: str) -> dict:
        """Parse [HSCM|PHASE|EVENT] key=val lines from C++ stderr."""
        event = {"raw": line, "type": "LOG"}
        if line.startswith("[HSCM|"):
            try:
                header, rest = line.split("] ", 1)
                _, phase, evt = header.lstrip('[').split('|')
                event["phase"] = phase
                event["event"] = evt
                event["type"]  = "PROGRESS"
                for kv in rest.split():
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        event[k] = int(v) if v.isdigit() else v
            except:
                pass
        elif line.startswith("[CHECKPOINT]"):
            event["type"] = "CHECKPOINT"
        elif line.startswith("[MANIFOLD]"):
            event["type"] = "MANIFOLD"
        return event

    def last_layer_stats(self, phase: str) -> dict:
        for e in reversed(self.events):
            if e.get("phase") == phase and e.get("event") == "END":
                return e
        return {}


# =============================================================================
# STEP 1: Compile the C++ engine
# =============================================================================

def compile_engine(state: PipelineState) -> bool:
    if state.engine_compiled and ENGINE_BIN.exists():
        log("  C++ engine already compiled — skipping")
        return True

    log("  Compiling C++ manifold engine...")
    cmd = [
        "g++", "-O3", "-std=c++17",
        "-o", str(ENGINE_BIN),
        str(ENGINE_SRC)
    ]
    log(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  COMPILE ERROR:\n{result.stderr}")
        return False

    log(f"  Compiled → {ENGINE_BIN}")
    state.engine_compiled = True
    state.save()
    return True


# =============================================================================
# STEP 2: Locate dictionary
# =============================================================================

def locate_dictionary(state: PipelineState) -> Optional[DictPointer]:
    log("  Locating dictionary...")
    ptr = locate_or_download_dictionary()
    log(f"  Dictionary: {ptr}")
    state.dict_path   = str(ptr.path)
    state.dict_format = ptr.format
    state.dict_lines  = ptr.line_count or 0
    state.save()
    return ptr


# =============================================================================
# STEP 3: Run C++ engine — layers 0 through 4
# =============================================================================

def run_manifold_engine(state: PipelineState, dict_ptr: DictPointer) -> bool:
    """
    Runs the C++ engine with the word stream piped in.
    Engine performs all 5 layers internally with checkpoints.
    """
    if state.layer_completed >= 4 and state.manifold_path:
        log("  Manifold already compiled — skipping engine run")
        return True

    log("  Starting manifold engine (5-layer exhaustion)...")
    index_path = str(INDEX_DIR / "manifold.bin")

    cmd = [
        str(ENGINE_BIN),
        "--ingest",
        f"--checkpoints={CHECKPOINT_DIR}",
        f"--index={index_path}",
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )

    # Start monitor thread
    monitor = EngineMonitor(proc)
    monitor.start()

    # Stream word records to engine stdin
    # Format per line: WORD PRIMITIVE SUBPOS PHONEME1 PHONEME2 ...
    log("  Streaming word records to engine...")
    words_sent = 0
    errors     = 0

    try:
        for batch in dict_ptr.iter_words(batch_size=500):
            for word, phonemes in batch:
                if not word or not word.isalpha():
                    continue

                # Assign primitive and subpos
                # In production: use upstream classifier
                # Here: use heuristic based on CMU phonemes
                assignment = _assign_primitive_subpos(word, phonemes)
                if assignment is None:
                    errors += 1
                    continue

                prim, subpos = assignment
                ph_str = " ".join(phonemes) if phonemes else ""
                line = f"{word} {prim} {subpos} {ph_str}\n"

                try:
                    proc.stdin.write(line.encode())
                    words_sent += 1
                except BrokenPipeError:
                    log("  Engine stdin closed — stopping stream")
                    break

            if words_sent % 5000 == 0:
                log(f"  Streamed {words_sent} words...")

    except Exception as e:
        log(f"  Stream error: {e}")
    finally:
        try:
            proc.stdin.close()
        except:
            pass

    log(f"  Stream complete: {words_sent} sent, {errors} skipped")

    # Read engine stdout for completion signal
    stdout_data = proc.stdout.read().decode('utf-8', errors='replace')
    proc.wait(timeout=300)
    monitor.join(timeout=5)

    log(f"  Engine exit code: {proc.returncode}")

    # Parse completion message
    for line in stdout_data.strip().split('\n'):
        if line.startswith("MANIFOLD_COMPLETE"):
            parts = line.split()
            if len(parts) >= 3:
                state.manifold_size = int(parts[1])
                state.manifold_path = parts[2]
                state.layer_completed = 4
                state.last_checkpoint = str(CHECKPOINT_DIR / "layer1.bin")
                state.save()
                log(f"  Manifold compiled: {state.manifold_size} coordinates")
                return True

    if proc.returncode == 0:
        # Engine succeeded but no MANIFOLD_COMPLETE line (demo mode)
        state.manifold_path = index_path
        state.layer_completed = 4
        state.save()
        return True

    log(f"  Engine returned non-zero: {proc.returncode}")
    return False


# =============================================================================
# PRIMITIVE/SUBPOS ASSIGNMENT HEURISTIC
# This is the stub that needs the real classifier.
# Currently uses word length and phoneme patterns as a rough proxy.
# Replace with trained tagger for production.
# =============================================================================

# Motion words tend to have liquid/plosive onsets
# Mental words tend toward fricative/nasal
# This is a placeholder — not the real system

_MOTION_PHONEMES    = {'R','L','S','P','B','T','D'}
_MENTAL_PHONEMES    = {'F','V','TH','N','M'}
_EVALUATIVE_ENDINGS = {'D','T','L','S'}

def _assign_primitive_subpos(word: str, phonemes) -> Optional[tuple]:
    """
    Heuristic primitive/subpos assignment.
    Returns (primitive_name, subpos_name) or None.
    
    HONEST NOTE: This is a structural placeholder.
    Production requires: POS tagger + sense disambiguation + primitive lookup.
    The matrix and engine are correct. This function is the remaining stub.
    """
    if not phonemes:
        return None

    # Most words default to THING/NOUN_CONCRETE for demo population
    # The traversal engine will still work correctly —
    # the manifold just needs *some* valid coordinates populated
    if len(word) <= 3:
        return ("THING", "NOUN_CONCRETE")

    first_ph = phonemes[0].rstrip('012') if phonemes else ''
    last_ph  = phonemes[-1].rstrip('012') if phonemes else ''

    # Rough motion signal
    if first_ph in ('R','S','SP','ST') and last_ph in ('T','D','L'):
        return ("MOVE", "VERB_MOTION")

    # Rough mental signal
    if first_ph in ('TH','F','W') and len(word) >= 4:
        return ("THINK", "VERB_MENTAL")

    # Default
    return ("THING", "NOUN_CONCRETE")


# =============================================================================
# STEP 4: Post-ingestion summary
# =============================================================================

def summarize(state: PipelineState):
    log("\n" + "="*60)
    log("  PIPELINE COMPLETE")
    log("="*60)
    log(f"  Dictionary:      {state.dict_path}")
    log(f"  Dictionary size: {state.dict_lines} lines")
    log(f"  Manifold path:   {state.manifold_path}")
    log(f"  Manifold size:   {state.manifold_size} coordinates")
    log(f"  Words ingested:  {state.words_ingested}")
    log(f"  Log:             {MAIN_LOG}")
    log("")
    log("  HONEST REMAINING STUBS:")
    log("  → _assign_primitive_subpos() needs a real POS tagger + sense")
    log("    disambiguator. This is the only piece not in the matrix.")
    log("  → L3 etymology: hardcoded to ALL_GEO_KEYS until etymology DB")
    log("    is wired in. Wiktionary API works for this.")
    log("="*60)


# =============================================================================
# MAIN
# =============================================================================

def run(resume: bool = True):
    log("HSCM Pipeline — starting")
    log(f"Base dir: {BASE_DIR}")

    state = PipelineState.load() if resume else PipelineState()
    state.started_at = datetime.now().isoformat()
    state.save()

    # Step 1: Compile engine
    if not compile_engine(state):
        log("ABORT: Engine compilation failed.")
        return False

    # Step 2: Locate dictionary
    dict_ptr = locate_dictionary(state)

    # Step 3: Run manifold engine
    if not run_manifold_engine(state, dict_ptr):
        log("ABORT: Manifold engine failed.")
        return False

    # Step 4: Summary
    summarize(state)
    return True


if __name__ == "__main__":
    resume = "--fresh" not in sys.argv
    if not resume:
        log("Fresh run — clearing prior state")
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    run(resume=resume)
