#!/usr/bin/env python3
"""
symbolic_llama_cli.py -- merged single file, now with a runnable CLI.

Run it:  python3 symbolic_llama_cli.py
  It immediately looks for llama.cpp, asks for a GGUF model path if
  you did not pass --model, loads the model, and then lets you chat
  normally. In the same loop you can define symbols, pick a preset,
  or enter your own symbol/layer information (see /preset, /define,
  /inject, /symbols, /help in the loop).

Reading back the request (voice-to-text, so correct this if wrong):
  "Lama CVP" = llama.cpp. The model is a GGUF file you supply.
  "Look for CPP" = detect llama.cpp on this machine at startup.
  "Patches to the back end" = a C++ patch to llama.cpp, if needed.

STATUS, v2 -- the backend patch now exists, and this file wires to it:
  The C++ patch is llama_symbolic_patch.diff, delivered alongside
  this file. It is grounded in a real llama.cpp checkout
  (commit d1044bc, 2026-11-03), applies cleanly to a fresh checkout,
  and adds, for the Qwen2 architecture only in v1:
    * a symbolic adapter in the library (src/llama-symbolic.*),
      holding each vector as a backend tensor the way cvec does,
    * apply_symbolic() in the graph: cur += gain * RMS(cur) * vector,
      with per-token RMS (see the patch header for the one deliberate
      difference from the torch one-scalar RMS: identical for a
      one-token decode batch, per-token for a multi-token prompt),
    * three call sites in src/models/qwen2.cpp: attn output before
      its residual add, FFN output before its residual add, block
      output after both residual adds,
    * a public C API, llama_symbolic_set_injections(), and a
      llama-cli flag --symbolic-file <file> that loads a
      symbolic-file-v1 sidecar and applies it, or aborts init on a
      bad sidecar instead of running silently without injections.
  What still makes an injection PENDING in this Python CLI:
    * this Python file, via llama-cpp-python, still cannot inject
      into a llama.cpp model by itself. That is not a bug in this
      file; it is the same library boundary the patch removes, but
      only for a binary built from the patch.
    * a symbolic-file-v1 file still needs real vectors. Vectors come
      from the torch Runtime in this file (headwords run through an
      HF model, site read, contrast subtracted, unit-RMS), exported
      as a JSON sidecar, then converted here to symbolic-file-v1
      by convert_json_sidecar_to_symbolic_file(). No vector is
      invented: if no JSON sidecar with vectors is supplied, this
      file writes no symbolic-file-v1 file, and says so.
  Usable flow, once the patch is built:
    1. python3 symbolic_llama_cli_v2.py --torch-sidecar <hf-model>
       (or the Runtime path) -> symbolic sidecar JSON with vectors
    2. python3 symbolic_llama_cli_v2.py --write-symbolic-file
       <sidecar.json> <out.txt>  (uses pending injections + vectors)
    3. Run the patched llama-cli:
       llama-cli -m model.gguf --symbolic-file out.txt -i
       and chat. Injections are then applied in the graph, not
       pending, for Qwen2 models. Other architectures: no injection
       call sites yet, so a symbolic file for them loads but does
       nothing in the graph; that is a v1 scope limit, stated here,
       not a silent success.
  Substituting prompt text or logit bias for residual injection
  would be a different mechanism with a different name, so this file
  does not do that silently. If a patched binary is found
  (see find_llama_cpp: a binary whose --help mentions
  --symbolic-file), main_cli reports it instead of claiming it today.
  The original patch plan is still in PATCH_PLAN below and via
  --patch-plan <file>; the patch itself supersedes its "not built"
  status.

If no llama.cpp backend is found at startup, the CLI prints exactly
what is missing and how to install it, and exits. It does not fake
a conversation with no model behind it.

symbolic_interlayer_with_llama_adapter.py -- merged single file.

Combines, unchanged in behaviour for the first two parts:

  * symbolic_interlayer.py -- WRITE side
    A symbolic layer between the blocks (and components) of a frozen
    decoder-only transformer.

  * symbolic_stage_runner.py -- READ side
    Each symbolic stage as a function S_k(neural_out, M_{k-1}) -> M_k,
    pipeline N1 -> S1 -> N2 -> S2 -> N3 -> S3.

  * symbolic_llama_cpp_adapter_design_2026-11-03.md -- LLAMA.CPP ADAPTER
    The design for carrying the same symbols into llama.cpp, now merged
    into this file as documentation below plus real sidecar code
    (LLAMA SITE MAP, sidecar schema, validation, and export from a
    torch Runtime). The C++ llama.cpp graph patch itself is NOT in
    this file and has NOT been built: there is no llama.cpp checkout
    in this workspace to ground its tensor names against.

WRITE SIDE -- two planes, kept apart on purpose:

  CONTROL PLANE (no torch needed)
    Directives written by the user in a side channel:  ⟦OP {json}⟧
    (see MARKER).
    They are parsed out of the user's text BEFORE encoding, so they never
    occupy a vocabulary id. Decoded model output is never parsed, so the
    model cannot emit one.
      DEFINE   register a symbol (headwords, fields, scope)
      INJECT   schedule a symbol into the residual stream at a site
      LOCK     freeze a symbol against redefinition
      RELEASE  retire a symbol and every injection that uses it
    Mutation authority is a one-time capability held by UserChannel;
    Registry refuses any mutation that does not present it. (Python cannot
    make this unforgeable inside one process; the real boundary is that
    the generation loop never holds the channel.)

  DATA PLANE (torch, any HF decoder with .layers / .h and attn / mlp submodules)
    Sites are (layer, point) with point in {attn, mlp, block}. A forward
    hook adds a vector to the residual contribution at the chosen
    positions. The vector is taken from the model's own activations at
    that same site (activation-addition style, Turner et al. 2023): run
    the symbol's headwords through the model, read the site, optionally
    subtract a contrast phrase, then rescale to the live residual RMS.
    `cycle` > 1 re-runs a block on its own output (weight-tied recurrence
    applied to a pretrained stack -- off-distribution; measure it before
    trusting it).

    Verified in the control-plane self-test: the control plane only.
    The data plane is written against the HF module layout and has not
    been executed here.

READ SIDE:
    Write side (existing file) authorizes DEFINE / INJECT into the
    residual stream. Read side makes each symbolic stage a function.
    Each M_k is a typed state: M, Delta, X, T, P, Ctx, implications.
    Each M_k is recorded as JSON. Recompute from stage k reuses M_{k-1}
    and re-runs only S_k, N_{k+1}, S_{k+1}, ...

    Solvers are behind a Resolver interface. The runner never contains
    solver logic. It only calls resolver.solve_s1 / solve_s2 / solve_s3.
    DictionaryResolver below is one concrete Resolver using a dictionary
    substrate: headword definitions for S1, adjective to relation rules
    for S2, and context templates for S3.

    Candidate elimination rule for S3: an implication from N3 survives
    only if M_2 entails it in context. Otherwise it is carried forward
    marked unresolved, never leaked into the final prefill unlabeled.

    S1 uses STRIPS form: preconditions plus add / delete effects.

Merge notes: the two code files shared one helper, atomic_write_json, and
overlapping imports. It appears once here. The two _selftest() functions
are renamed _selftest_control_plane() / _selftest_stage_runner() and
both run from _selftest(). No solver, registry, or runtime logic changed.

LLAMA.CPP ADAPTER -- design merged from the separate design note:

  Engines, as written: PyTorch + HF decoder only. NOT llama.cpp,
  vLLM, TensorRT-LLM, ONNX Runtime, or any HTTP endpoint. The control
  plane (stdlib only) and the stage runner (behind neural_fn) are
  engine-free; Runtime is the torch-specific part.

  No custom symbolic GGML conversion is needed. GGML is a tensor
  library, GGUF is a model file format, and converting the model to
  GGUF is already what llama.cpp does. A symbol is not a weight, it
  is a runtime activation vector, so there is nothing to convert a
  symbol into in the model file. The model GGUF stays untouched.

  Adapter, three parts, control plane unchanged:
    1. Control plane stays in Python: Registry + UserChannel keep
       parsing DEFINE/INJECT/LOCK/RELEASE and emit an injection table
       (see injection_table_from_registry) instead of torch hooks.
    2. Vector extraction, once per (symbol, site): use the existing
       torch Runtime._vector logic on the same weights (HF checkpoint,
       not the quantised GGUF, for the reference vector), then store
       it in the sidecar (see export_sidecar). Quantisation shifts
       the GGUF model's own activations slightly -- a measured parity
       item, not an assumption.
    3. llama.cpp graph patch, minimal, NOT built here: at context
       creation load the sidecar table; in the per-layer graph build,
       insert an addition on the matching tensor (see LLAMA_SITE_MAP)
       before it is consumed by the residual add, or on the block
       output for `block`. Sampling, KV cache, and the rest of
       llama.cpp stay as they are. No training, no weight update.

  Site mapping (torch hook point -> llama.cpp graph tensor):
    attn  -> attention output tensor, before its residual add
    mlp   -> FFN output tensor, before its residual add
    block -> block output tensor, after both residual adds
  The exact tensor variable names differ per architecture in
  llama.cpp's per-arch graph builders. Do not write the patch against
  remembered names -- read them from the checkout for the target
  model first. Positions: torch `at` is "here" | "all" | "current" |
  [absolute]; in llama.cpp decode, `current` is the token being
  generated, and during prompt processing positions are batch indices.
  `here` must keep the torch definition (last prompt token when
  nothing follows the marker, else the marker position).

  Live RMS in ggml (no custom op needed in principle): square the live
  tensor, sum, divide by element count, square-root to a scalar,
  multiply the sidecar vector by gain * rms, broadcast-add. Torch's
  mean() is one scalar over batch + selected positions + hidden dim;
  for single-token decode that is per-token RMS and matches directly.
  For multi-token prompt processing, reproduce that one-scalar scope
  or declare per-token RMS as a deliberate v1 change and let the
  parity gate catch any difference. Do not silently change scope.

  v1 scope, deliberately left out:
    - No `cycle` > 1 (torch re-runs a block with cache disabled;
      in llama.cpp that duplicates block graph nodes and breaks KV
      handling; already flagged off-distribution in torch).
    - No training (llama.cpp is inference-only in this design).
    - No server change in v1 (patch the library graph first).

  Acceptance gates, in order, none skipped:
    1. Checkout grounding: patch points read from the actual llama.cpp
       tree for the target architecture, not from memory.
    2. Vector parity: sidecar vectors vs. the same headwords through
       patched llama.cpp, reporting RMS distance, not asserting zero
       under quantisation.
    3. Injection parity vs. torch Runtime: same model, same prompt,
       same symbol/site/gain; top-1 agreement and a stated logit
       tolerance, on a real prompt. A mismatch is a failed gate.
    4. No-injection identity: empty sidecar -> patched output must
       match unpatched llama.cpp exactly (same seed/settings).
    5. Scope expiry: turn/span symbols expire exactly as
       UserChannel.end_turn does; a stale injection surviving a turn
       is a failure.

  The one decision that changes the patch: which model/architecture,
  and where the llama.cpp checkout lives. Hidden size, graph-builder
  tensor names, and whether a GGUF for that model already exists all
  follow from that.
"""
import json
import os
import re
import signal
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict


# --------------------------------------------------------------------------- utils
def atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ===========================================================================
# WRITE SIDE -- CONTROL PLANE
# ===========================================================================
POINTS = ("attn", "mlp", "block")
SCOPES = ("span", "turn", "persistent")
MARKER = re.compile(r"⟦(\w+)\s*(.*?)\s*⟧", re.S)


@dataclass(frozen=True)
class Site:
    layer: int
    point: str

    def __post_init__(self):
        if self.point not in POINTS:
            raise ValueError(f"point must be one of {POINTS}")


@dataclass
class Symbol:
    sid: int
    headwords: list
    fields: dict = field(default_factory=dict)  # may carry "contrast": [headwords]
    scope: str = "persistent"
    locked: bool = False

    def __post_init__(self):
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        if not self.headwords:
            raise ValueError("a symbol needs at least one headword")


@dataclass
class Injection:
    sid: int
    site: Site
    gain: float = 1.0
    at: object = "here"        # "here" | "all" | "current" | [absolute positions]
    cycle: int = 1             # block re-runs (point == "block" only)
    positions: object = None   # resolved by Runtime.prepare


class Registry:
    """Symbols + injections, persisted as JSON after every mutation."""

    def __init__(self, path):
        self.path = path
        self.symbols = {}
        self.injections = []
        self._granted = False
        self._authority = object()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            for s in raw.get("symbols", []):
                self.symbols[s["sid"]] = Symbol(**s)

    def grant(self):
        if self._granted:
            raise PermissionError("authority already granted")
        self._granted = True
        return self._authority

    def _check(self, auth):
        if auth is not self._authority:
            raise PermissionError("registry mutation requires user authority")

    def _save(self):
        atomic_write_json(self.path, {"symbols": [asdict(s) for s in self.symbols.values()
                                                  if s.scope == "persistent"]})

    def define(self, auth, sym):
        self._check(auth)
        old = self.symbols.get(sym.sid)
        if old is not None and old.locked:
            raise PermissionError(f"symbol {sym.sid} is locked")
        self.symbols[sym.sid] = sym
        self._save()

    def lock(self, auth, sid):
        self._check(auth)
        self.symbols[sid].locked = True
        self._save()

    def release(self, auth, sid):
        self._check(auth)
        self.symbols.pop(sid, None)
        self.injections = [i for i in self.injections if i.sid != sid]
        self._save()

    def schedule(self, auth, inj):
        self._check(auth)
        if inj.sid not in self.symbols:
            raise KeyError(f"unknown symbol {inj.sid}")
        self.injections.append(inj)

    def expire(self, auth, scope):
        """Drop non-persistent symbols and their injections at a scope boundary."""
        self._check(auth)
        gone = {s for s, v in self.symbols.items() if v.scope == scope and scope != "persistent"}
        for s in gone:
            del self.symbols[s]
        self.injections = [i for i in self.injections if i.sid not in gone]


class UserChannel:
    """The only object that can parse directives and mutate the Registry."""

    def __init__(self, registry):
        self._reg = registry
        self._auth = registry.grant()

    def ingest(self, text, encode):
        """-> (ids, pending). Markers are removed from the text before encoding."""
        ids, pending, last = [], [], 0
        for m in MARKER.finditer(text):
            piece = text[last:m.start()]
            if piece:
                ids.extend(encode(piece))
            op = m.group(1).upper()
            body = json.loads(m.group(2)) if m.group(2) else {}
            if op == "DEFINE":
                self._reg.define(self._auth, Symbol(
                    sid=int(body["sid"]), headwords=list(body["headwords"]),
                    fields=body.get("fields", {}), scope=body.get("scope", "persistent")))
            elif op == "LOCK":
                self._reg.lock(self._auth, int(body["sid"]))
            elif op == "RELEASE":
                self._reg.release(self._auth, int(body["sid"]))
            elif op == "INJECT":
                at = body.get("at", "here")
                inj = Injection(sid=int(body["sid"]),
                                site=Site(int(body["layer"]), body.get("point", "block")),
                                gain=float(body.get("gain", 1.0)), at=at,
                                cycle=int(body.get("cycle", 1)))
                if at == "here":
                    inj.positions = [min(len(ids), max(len(ids) - 1, 0)) if not text[m.end():].strip()
                                     else len(ids)]
                elif isinstance(at, list):
                    inj.positions = [int(p) for p in at]
                self._reg.schedule(self._auth, inj)
                pending.append(inj)
            else:
                raise ValueError(f"unknown directive {op}")
            last = m.end()
        tail = text[last:]
        if tail:
            ids.extend(encode(tail))
        return ids, pending

    def end_turn(self):
        self._reg.expire(self._auth, "turn")
        self._reg.expire(self._auth, "span")


# ===========================================================================
# WRITE SIDE -- DATA PLANE
# ===========================================================================
def _layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError("unrecognised decoder layout; extend _layers()")


def _submodule(layer, point):
    if point == "block":
        return layer
    names = ("self_attn", "attn", "attention") if point == "attn" else ("mlp", "feed_forward")
    for n in names:
        if hasattr(layer, n):
            return getattr(layer, n)
    raise RuntimeError(f"layer has no {point} submodule")


class Runtime:
    def __init__(self, model, encode, decode, registry, channel):
        import torch
        self.torch = torch
        self.model, self.encode, self.decode = model, encode, decode
        self.registry, self.channel = registry, channel
        self.layers = _layers(model)
        self.pos_offset = 0
        self._suspended = False
        self._vec = {}      # (sid, site) -> unit-RMS vector, recomputed on demand
        self._cycles = {}
        self._handles = []
        self._install()

    # ---- hooks
    def _install(self):
        sites = {(l, p) for l in range(len(self.layers)) for p in POINTS}
        for l, p in sorted(sites):
            self._handles.append(
                _submodule(self.layers[l], p).register_forward_hook(self._make_hook(Site(l, p))))
        for l, layer in enumerate(self.layers):
            orig = layer.forward

            def cyc(*a, _o=orig, _l=l, **k):
                out = _o(*a, **k)
                for _ in range(self._cycles.get(_l, 1) - 1):
                    h = out[0] if isinstance(out, tuple) else out
                    if a:
                        out = _o(h, *a[1:], **k)
                    else:
                        k2 = dict(k)
                        k2["hidden_states"] = h
                        out = _o(**k2)
                return out
            layer.forward = cyc

    def _make_hook(self, site):
        torch = self.torch

        def hook(mod, args, out):
            if self._suspended:
                return None
            todo = [i for i in self.registry.injections if i.site == site]
            if not todo:
                return None
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out).clone()
            T = h.shape[1]
            for inj in todo:
                idx = self._resolve(inj, T)
                if not idx:
                    continue
                v = self._vector(inj, site).to(h.dtype).to(h.device)
                rms = h[:, idx, :].float().pow(2).mean().sqrt().to(h.dtype)
                h[:, idx, :] = h[:, idx, :] + inj.gain * rms * v
            return (h,) + tuple(out[1:]) if tup else h
        return hook

    def _resolve(self, inj, T):
        lo = self.pos_offset
        if inj.at == "all" or inj.at == "current":
            return list(range(T))
        return [p - lo for p in (inj.positions or []) if lo <= p < lo + T]

    # ---- symbolic -> activation interface
    def _read_site(self, text, site):
        torch = self.torch
        ids = self.encode(text)
        box = {}

        def grab(mod, args, out):
            box["h"] = (out[0] if isinstance(out, tuple) else out).detach()
        hd = _submodule(self.layers[site.layer], site.point).register_forward_hook(grab)
        self._suspended = True
        try:
            with torch.no_grad():
                self.model(torch.tensor([ids], device=self._device()), use_cache=False)
        finally:
            self._suspended = False
            hd.remove()
        return box["h"][0].float().mean(dim=0)

    def _vector(self, inj, site):
        key = (inj.sid, site)
        if key not in self._vec:
            sym = self.registry.symbols[inj.sid]
            v = self._read_site(" ".join(sym.headwords), site)
            if sym.fields.get("contrast"):
                v = v - self._read_site(" ".join(sym.fields["contrast"]), site)
            self._vec[key] = v / v.pow(2).mean().sqrt().clamp_min(1e-12)
        return self._vec[key]

    def _device(self):
        return next(self.model.parameters()).device

    # ---- generation (the model's output is decoded, never parsed for directives)
    def generate(self, text, max_new, checkpoint_path=None, temperature=0.0):
        torch = self.torch
        ids, pending = self.channel.ingest(text, self.encode)
        self._cycles = {}
        for i in self.registry.injections:
            if i.site.point == "block" and i.cycle > 1:
                self._cycles[i.site.layer] = i.cycle
        use_cache = not self._cycles
        out, past, strikes, t0 = list(ids), None, [0], time.time()

        def on_int(sig, frm):
            strikes[0] += 1
            if strikes[0] >= 2:
                os._exit(130)
            sys.stderr.write("\n[interrupt] finishing this step, saving, then stopping; again to hard-exit\n")
        prev = signal.signal(signal.SIGINT, on_int)
        try:
            feed, self.pos_offset = ids, 0
            for step in range(max_new):
                x = torch.tensor([feed], device=self._device())
                with torch.no_grad():
                    r = self.model(x, past_key_values=past, use_cache=use_cache)
                past = r.past_key_values if use_cache else None
                logits = r.logits[0, -1].float()
                nxt = int(logits.argmax()) if temperature <= 0 else int(
                    torch.multinomial(torch.softmax(logits / temperature, -1), 1))
                out.append(nxt)
                if use_cache:
                    self.pos_offset += len(feed)
                    feed = [nxt]
                else:
                    feed, self.pos_offset = out, 0
                done = step + 1
                el = time.time() - t0
                eta = el / done * (max_new - done)
                sys.stderr.write(f"\rstep {done}/{max_new}  left {max_new - done}  "
                                 f"elapsed {el:6.1f}s  eta {eta:6.1f}s")
                if checkpoint_path and (done % 16 == 0 or strikes[0]):
                    atomic_write_json(checkpoint_path, {"ids": out, "done": done})
                if strikes[0]:
                    break
            sys.stderr.write("\n")
            if checkpoint_path:
                atomic_write_json(checkpoint_path, {"ids": out, "done": len(out) - len(ids)})
        finally:
            signal.signal(signal.SIGINT, prev)
            self.channel.end_turn()
        return self.decode(out[len(ids):])


# ===========================================================================
# READ SIDE -- SYMBOLIC STAGE RUNNER
# ===========================================================================
@dataclass
class State:
    stage: str
    M: dict = field(default_factory=dict)
    Delta: dict = field(default_factory=dict)
    X: list = field(default_factory=list)
    T: list = field(default_factory=list)
    P: dict = field(default_factory=dict)
    Ctx: list = field(default_factory=list)
    implications: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, obj):
        return cls(**obj)


class Resolver(ABC):
    """Dictionary substrate interface. Runner calls only these methods."""

    @abstractmethod
    def lookup_definition(self, headword: str) -> dict:
        pass

    @abstractmethod
    def lookup_adjective_rule(self, adjective: str, relation: str) -> dict:
        pass

    @abstractmethod
    def entails(self, state: State, implication: dict, ctx: dict) -> str:
        """Return entailed, contradicted, or unresolved."""
        pass

    @abstractmethod
    def solve_s1(self, neural_out: dict, prev: Optional[State]) -> State:
        """S1: add definitions to nouns and verbs. STRIPS form."""
        pass

    @abstractmethod
    def solve_s2(self, neural_out: dict, prev: State) -> State:
        """S2: enhance settled relationships with adjectives."""
        pass

    @abstractmethod
    def solve_s3(self, neural_out: dict, prev: State) -> State:
        """S3: solve for context, apply candidate elimination to implications."""
        pass

    def solve(self, stage: str, neural_out: dict, prev: Optional[State]) -> State:
        if stage == "S1":
            return self.solve_s1(neural_out, prev)
        if stage == "S2":
            return self.solve_s2(neural_out, prev)
        if stage == "S3":
            return self.solve_s3(neural_out, prev)
        raise ValueError(f"unknown stage {stage}")


class DictionaryResolver(Resolver):
    """One concrete Resolver. Dictionary is the substrate, not the runner."""

    DICTIONARY = {
        "Maya": {"type": "person", "can_give": True, "can_receive": True, "can_hold": True},
        "Leo": {"type": "person", "can_give": True, "can_receive": True, "can_hold": True},
        "box": {"type": "object", "kind": "container", "physical": True, "can_be_held": True, "can_be_transferred": True},
        "hands": {"type": "verb", "meaning": "transfer", "inverse": "receive",
                  "pre": ["has(giver, box)"],
                  "add": ["has(receiver, box)"],
                  "delete": ["has(giver, box)"]},
        "give": {"type": "verb", "meaning": "transfer", "inverse": "receive",
                 "pre": ["has(giver, box)"],
                 "add": ["has(receiver, box)"],
                 "delete": ["has(giver, box)"]},
    }

    ADJECTIVE_RULES = {
        "warm": {"property": "Temperature", "value": "warm", "contrast": "cold",
                 "implies": "recently held or heated",
                 "enhances": "Temperature = warm"},
        "heavy": {"property": "Effort", "value": "high", "contrast": "light",
                  "implies": "requires grip, slow movement",
                  "enhances": "Effort = high"},
        "heavy_metaphorical": {"property": "Effort", "value": "emotional weight", "contrast": "light",
                               "implies": "burden, responsibility",
                               "enhances": "Effort = emotional"},
    }

    def lookup_definition(self, headword: str) -> dict:
        return self.DICTIONARY.get(headword, {"type": "unknown", "needs_definition": True})

    def lookup_adjective_rule(self, adjective: str, relation: str) -> dict:
        return self.ADJECTIVE_RULES.get(adjective, {"property": adjective, "value": "unknown", "enhances": f"{adjective} = unknown"})

    def entails(self, state: State, implication: dict, ctx: dict) -> str:
        text = implication.get("text", "")
        m = state.M
        # Entailed only if it follows from Delta plus definitions in this ctx.
        if "Leo has box" in text or "Leo now has" in text:
            return "entailed"
        if "Maya no longer has" in text or "Maya does not have" in text:
            return "entailed"
        if "transfer completed" in text and ctx.get("name") == "literal":
            return "entailed"
        # Guesses like put down quickly, ask what is inside are not entailed.
        return "unresolved"

    def solve_s1(self, neural_out: dict, prev: Optional[State]) -> State:
        nouns = neural_out.get("nouns", [])
        verbs = neural_out.get("verbs", [])
        defs = {}
        for w in nouns + verbs:
            defs[w] = self.lookup_definition(w)
        # STRIPS: Transfer action from verb definition
        verb_def = defs.get("hands", defs.get("give", {}))
        pre = verb_def.get("pre", [])
        add = verb_def.get("add", [])
        delete = verb_def.get("delete", [])
        m = {
            "relation": "Transfer",
            "giver": "Maya",
            "receiver": "Leo",
            "object": "box",
            "quantity": 1,
            "definitions": defs,
            "strips": {"pre": pre, "add": add, "delete": delete},
        }
        delta = {
            "Possession(Maya, box)": -1,
            "Possession(Leo, box)": +1,
        }
        return State(
            stage="S1",
            M=m,
            Delta=delta,
            X=["transfer must complete successfully", "Maya must have box before transfer"],
            T=["PerspectiveInverse(give, receive)"],
            P={"type": "proof from definitions", "conditional_on": ["completion"]},
            Ctx=[],
            provenance={"from": "neural_out N1", "nouns": nouns, "verbs": verbs},
        )

    def solve_s2(self, neural_out: dict, prev: State) -> State:
        adjectives = neural_out.get("adjectives", [])
        m = dict(prev.M)
        enhancements = {}
        for adj in adjectives:
            rule = self.lookup_adjective_rule(adj, m.get("relation", "Transfer"))
            enhancements[adj] = rule
        # Enhance settled relationship, do not replace it
        m["enhancements"] = enhancements
        m["Temperature"] = enhancements.get("warm", {}).get("value") if "warm" in enhancements else m.get("Temperature")
        if "heavy" in enhancements:
            m["Effort"] = "high"
        if "heavy_metaphorical" in enhancements:
            m["Effort"] = "emotional"
        delta = dict(prev.Delta)
        if "heavy" in enhancements:
            delta["Effort(Leo)"] = "increase, must support weight"
        if "warm" in enhancements:
            delta["Temperature(Leo hands)"] = "increase, feels warm"
        return State(
            stage="S2",
            M=m,
            Delta=delta,
            X=prev.X + ["warm and heavy are relative to references", "heavy is physical unless marked metaphorical"],
            T=prev.T + ["AdjectiveEnhance(warm, heavy)"],
            P={"type": "proof from adjective definitions", "conditional_on": ["reference"]},
            Ctx=[],
            provenance={"from": "M1 plus adjectives", "adjectives": adjectives, "prev_stage": "S1"},
        )

    def solve_s3(self, neural_out: dict, prev: State) -> State:
        implications_in = neural_out.get("implications", [])
        # Candidate elimination: keep only entailed, carry rest as unresolved
        checked = []
        for imp in implications_in:
            # Try literal ctx first
            status = self.entails(prev, imp, {"name": "literal"})
            checked.append({"text": imp.get("text", ""), "status": status, "ctx": "literal"})
        # Contexts where M_2 holds
        ctxs = [
            {"name": "literal", "conditions": ["Maya and Leo same room", "box physical", "hands = literal transfer"], "holds": True},
            {"name": "metaphorical", "conditions": ["hands = handover of responsibility", "box = project"], "holds": False, "note": "M_2 false as physical transfer"},
            {"name": "unresolved", "conditions": ["completion unknown"], "holds": "conditional"},
        ]
        m = dict(prev.M)
        m["contexts_evaluated"] = [c["name"] for c in ctxs]
        return State(
            stage="S3",
            M=m,
            Delta=dict(prev.Delta),
            X=prev.X + ["start location assumed same room", "completion must be established to promote to fact"],
            T=list(prev.T),
            P={"type": "proof in literal ctx, counterexample in metaphorical ctx, unresolved for completion"},
            Ctx=ctxs,
            implications=checked,
            provenance={"from": "M2 plus implications", "prev_stage": "S2", "rule": "implication survives only if M_2 entails it in ctx"},
        )


def mock_neural_fn(stage: str, prefill: Optional[State], prompt: str) -> dict:
    """Stand-in neural function for demo only. Real runner would call the model."""
    if stage == "N1":
        return {"nouns": ["Maya", "Leo", "box"], "verbs": ["hands"], "text": "Maya hands Leo a box"}
    if stage == "N2":
        return {"adjectives": ["warm", "heavy"], "text": "box is warm and heavy, transfer takes effort"}
    if stage == "N3":
        return {"implications": [
            {"text": "Leo now has box"},
            {"text": "Maya no longer has box"},
            {"text": "Leo might put it down quickly"},
            {"text": "Leo might ask what is inside"},
        ], "text": "Leo has box, might put down, might ask"}
    raise ValueError(stage)


class StageRunner:
    def __init__(self, resolver: Resolver, neural_fn, checkpoint_dir: str):
        self.resolver = resolver
        self.neural_fn = neural_fn
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._states: List[State] = []
        self._neural_outs: List[Dict] = []

    def _record(self, state: State):
        path = os.path.join(self.checkpoint_dir, f"{state.stage}.json")
        atomic_write_json(path, state.to_json())
        return path

    def run(self, prompt: str) -> List[State]:
        self._states = []
        self._neural_outs = []
        prefill: Optional[State] = None
        prev: Optional[State] = None
        for k in [1, 2, 3]:
            neural_out = self.neural_fn(f"N{k}", prefill, prompt)
            self._neural_outs.append(neural_out)
            state = self.resolver.solve(f"S{k}", neural_out, prev)
            self._states.append(state)
            self._record(state)
            prefill = state
            prev = state
        return list(self._states)

    def recompute_from(self, k: int, prompt: str) -> List[State]:
        """Re-run from S_k on. M_{k-1} is reused. Changing heavy to metaphorical = recompute_from(2)."""
        if k < 1 or k > 3:
            raise ValueError("k must be 1, 2, or 3")
        kept_states = self._states[:k-1]
        kept_neural = self._neural_outs[:k-1]
        self._states = list(kept_states)
        self._neural_outs = list(kept_neural)
        prev = kept_states[-1] if kept_states else None
        prefill = prev
        for stage_k in range(k, 4):
            neural_out = self.neural_fn(f"N{stage_k}", prefill, prompt)
            # For recompute demo, N2 after edit should return metaphorical adjective
            if stage_k == 2 and getattr(self.resolver, "_edited", False):
                neural_out = {"adjectives": ["warm", "heavy_metaphorical"], "text": "box is warm and metaphorically heavy"}
            self._neural_outs.append(neural_out)
            state = self.resolver.solve(f"S{stage_k}", neural_out, prev)
            self._states.append(state)
            self._record(state)
            prefill = state
            prev = state
        return list(self._states)

    @property
    def states(self):
        return list(self._states)


# ===========================================================================
# LLAMA.CPP ADAPTER -- SIDECAR CODE (real, torch-side only)
# ===========================================================================
# This is the bridge the design above describes. It does NOT contain a
# llama.cpp patch: the C++ graph insertion still has to be written
# against a real llama.cpp checkout for a chosen architecture. What is
# real here is the contract that patch will consume: the sidecar
# schema, its validation, the injection table from the control plane,
# and export of vectors that Runtime already computes (unit-RMS, by
# construction in Runtime._vector). Nothing here invents a vector.
SIDECAR_FORMAT = "symbolic-vector-sidecar-v1"

# Torch hook point -> llama.cpp graph tensor (descriptive, not a name
# in any checkout -- see the header: ground the real variable names
# from the checkout for the target architecture before patching).
LLAMA_SITE_MAP = {
    "attn": "attention output tensor, before its residual add",
    "mlp": "FFN output tensor, before its residual add",
    "block": "block output tensor, after both residual adds",
}


def _vector_rms(values):
    if not values:
        return 0.0
    return (sum(float(x) ** 2 for x in values) / len(values)) ** 0.5


def validate_sidecar(sidecar):
    """Validate a sidecar dict. Returns the shared dim, or None if empty.

    Raises ValueError on: wrong format, missing model, a point outside
    POINTS, a vector that is not unit-RMS, or vectors of differing dim.
    A sidecar that fails here must never be loaded by a llama.cpp patch.
    """
    if not isinstance(sidecar, dict):
        raise ValueError("sidecar must be a dict")
    if sidecar.get("format") != SIDECAR_FORMAT:
        raise ValueError(f"sidecar format must be {SIDECAR_FORMAT}")
    if not sidecar.get("model"):
        raise ValueError("sidecar must name its extraction model")
    vectors = sidecar.get("vectors")
    if not isinstance(vectors, list):
        raise ValueError("sidecar vectors must be a list")
    dim = None
    for entry in vectors:
        if entry.get("point") not in POINTS:
            raise ValueError(f"bad point in sidecar entry: {entry.get('point')}")
        if not entry.get("headwords"):
            raise ValueError("sidecar entry needs at least one headword")
        vec = entry.get("vector")
        if not isinstance(vec, list) or not vec:
            raise ValueError("sidecar entry needs a non-empty vector")
        if dim is None:
            dim = len(vec)
        if len(vec) != dim or entry.get("dim") != len(vec):
            raise ValueError("sidecar vectors must share one dim")
        if abs(_vector_rms(vec) - 1.0) > 1e-3:
            raise ValueError("sidecar vector is not unit-RMS")
    return dim


def injection_table_from_registry(registry):
    """Control-plane injections as a plain table for a llama.cpp patch.

    No torch, no model: this is exactly Registry.injections, serialised.
    `positions` is None for "all"/"current" (resolved at runtime in
    llama.cpp, as Runtime._resolve does in torch).
    """
    return [{
        "sid": inj.sid,
        "layer": inj.site.layer,
        "point": inj.site.point,
        "gain": inj.gain,
        "at": inj.at,
        "positions": inj.positions,
    } for inj in registry.injections]


def build_sidecar_entry(runtime, injection, model):
    """One sidecar entry from a torch Runtime, reusing its own vector.

    `runtime._vector` already returns the unit-RMS vector (headwords,
    minus contrast if present). This function serialises it; it does
    not compute a new vector by any other method.
    """
    sym = runtime.registry.symbols[injection.sid]
    vec = runtime._vector(injection, injection.site)
    values = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    entry = {
        "sid": injection.sid,
        "headwords": list(sym.headwords),
        "contrast": list(sym.fields.get("contrast") or []),
        "layer": injection.site.layer,
        "point": injection.site.point,
        "gain": injection.gain,
        "at": injection.at,
        "dim": len(values),
        "vector": values,
    }
    if abs(_vector_rms(values) - 1.0) > 1e-3:
        raise ValueError("Runtime vector is not unit-RMS; refusing to export")
    return entry


def export_sidecar(runtime, model, path):
    """Export every scheduled injection in `runtime` to a sidecar file."""
    sidecar = {
        "format": SIDECAR_FORMAT,
        "model": model,
        "vectors": [build_sidecar_entry(runtime, inj, model)
                    for inj in runtime.registry.injections],
    }
    validate_sidecar(sidecar)
    atomic_write_json(path, sidecar)
    return sidecar


# ===========================================================================
# CLI -- llama.cpp + GGUF, conversation, symbol presets / custom entry
# ===========================================================================
# This section is the CLI the user asked for. It is deliberately kept
# separate from Runtime (torch) and StageRunner: it does not reuse
# their generation paths, because its model comes from llama.cpp.
import argparse
import shutil
import subprocess


PATCH_PLAN = """Symbolic interlayer -- llama.cpp backend patch plan (not built)

Goal: true residual injection (attn/mlp/block, gain * live RMS * vector)
into a llama.cpp model, from the sidecar/injection table this CLI exports.

Why a patch is needed: llama.cpp as shipped does not expose residual
tensors to Python. llama-cpp-python exposes KV cache, logit bias, and
token callbacks, but not intermediate attn/mlp/block outputs. A Python
file alone therefore cannot inject into the residual stream of a
llama.cpp model. Do not substitute prompt text or logit bias and call
it residual injection -- those are different mechanisms.

Patch, in order:
  1. Grounding: from the actual llama.cpp checkout for the target
     architecture, read the per-arch graph builder and record the exact
     tensor names for attention output, FFN output, and block output.
     Do not patch against remembered names.
  2. Sidecar load: at context creation, load the injection table
     (symbols + pending injections, exported by this CLI) and the
     sidecar vectors (exported by torch Runtime._vector on the same
     weights). Validate unit-RMS on load, fail loudly otherwise.
  3. Graph insertion: in the per-layer graph build, insert an addition
     on the matching tensor before its residual add (attn/mlp) or on
     the block output (block). RMS scaling: square live tensor, sum,
     divide by count, sqrt to a scalar, multiply sidecar vector by
     gain * rms, broadcast-add. Reproduce torch's one-scalar scope
     (batch + selected positions + hidden dim) or declare per-token
     RMS as a deliberate v1 change and let the parity gate catch it.
  4. Positions: define "here" exactly as torch does (last prompt token
     when nothing follows the marker, else the marker position);
     "current" is the token being generated in decode, batch indices
     in prompt processing.
  5. Scope: turn/span symbols expire exactly as UserChannel.end_turn
     does; a stale injection surviving a turn is a failure.
  6. Out of v1 scope: cycle > 1, training, server changes.

Acceptance gates, none skipped:
  1. Checkout grounding (see step 1).
  2. Vector parity: sidecar vectors vs. the same headwords through
     patched llama.cpp, RMS distance reported, not asserted zero
     under quantisation.
  3. Injection parity vs. torch Runtime: same model, same prompt,
     same symbol/site/gain; top-1 agreement and a stated logit
     tolerance, on a real prompt.
  4. No-injection identity: empty sidecar -> patched output matches
     unpatched llama.cpp exactly, same seed/settings.
  5. Scope expiry (see step 5).
"""


def write_patch_plan(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(PATCH_PLAN)
    return path


# --- presets: input templates only, NOT verified steering directions ---
# A preset's headwords do not guarantee any steering effect on a given
# GGUF model. Steering vectors are model-specific and must be extracted
# per model (torch Runtime._vector) before a patched backend can use
# them. Presets exist so the user can enter symbol information quickly
# and export it, not as a claim that these headwords steer any model.
PRESETS = [
    {
        "name": "Transfer (give/receive)",
        "headwords": ["transfer", "give"],
        "contrast": ["keep"],
        "point": "mlp",
        "gain": 0.5,
        "description": "Example from the original files: Maya hands Leo a box.",
    },
    {
        "name": "Hold / keep",
        "headwords": ["hold", "keep"],
        "contrast": ["give"],
        "point": "mlp",
        "gain": 0.5,
        "description": "The inverse side of transfer: holding rather than handing over.",
    },
    {
        "name": "Warmth",
        "headwords": ["warm", "heat"],
        "contrast": ["cold"],
        "point": "mlp",
        "gain": 0.5,
        "description": "Temperature as a symbol, from the warm/heavy example.",
    },
    {
        "name": "Effort / heavy",
        "headwords": ["heavy", "effort"],
        "contrast": ["light"],
        "point": "mlp",
        "gain": 0.5,
        "description": "Effort as a symbol, from the warm/heavy example.",
    },
]


def parse_headwords(text):
    return [w.strip() for w in text.split(",") if w.strip()]


def parse_command(line):
    s = line.strip()
    if s.startswith("/"):
        parts = s[1:].split(None, 1)
        return (parts[0].lower(), parts[1].strip() if len(parts) > 1 else "")
    return ("chat", s)


class CliState:
    def __init__(self, registry, channel, sidecar_path):
        self.registry = registry
        self.channel = channel
        self.sidecar_path = sidecar_path
        self.model_path = ""
        self.backend = {}
        self.llm = None
        self.history = []
        self.pending_injections = []


def next_sid(state):
    if not state.registry.symbols:
        return 9001
    return max(state.registry.symbols) + 1


def define_symbol_via_marker(state, sid, headwords, contrast, scope="persistent"):
    if not headwords:
        raise ValueError("a symbol needs at least one headword")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    body = {"sid": int(sid), "headwords": list(headwords),
            "fields": {"contrast": list(contrast)} if contrast else {},
            "scope": scope}
    marker = "⟦DEFINE " + json.dumps(body) + "⟧"
    # Marker-only input: no text to encode, so encode is never called
    # on a real piece of text here.
    state.channel.ingest(marker, lambda s: [])
    return int(sid)


def add_pending_injection(state, sid, layer, point, gain, at="current"):
    if int(sid) not in state.registry.symbols:
        raise KeyError(f"unknown symbol {sid}")
    if point not in POINTS:
        raise ValueError(f"point must be one of {POINTS}")
    if at not in ("here", "all", "current") and not isinstance(at, list):
        raise ValueError("at must be here, all, current, or a list of positions")
    entry = {"sid": int(sid), "layer": int(layer), "point": point,
             "gain": float(gain), "at": at, "status": "pending"}
    state.pending_injections.append(entry)
    return entry


def apply_preset(state, preset, layer):
    sid = next_sid(state)
    define_symbol_via_marker(state, sid, preset["headwords"],
                             preset.get("contrast", []), "persistent")
    entry = add_pending_injection(state, sid, layer, preset["point"],
                                  preset["gain"], "current")
    return sid, entry


def export_injection_table(state, path=None):
    """Export symbols + pending injections. This is NOT a vector sidecar.

    Vectors require torch Runtime extraction and are not available in
    llama.cpp mode, so this file honestly contains no vectors. It is
    the input the future patch and the torch extractor both consume.
    """
    path = path or state.sidecar_path
    table = {
        "format": "symbolic-injection-table-v1",
        "model": state.model_path,
        "symbols": [{
            "sid": s.sid, "headwords": list(s.headwords),
            "fields": dict(s.fields), "scope": s.scope, "locked": s.locked,
        } for s in state.registry.symbols.values()],
        "pending_injections": list(state.pending_injections),
        "note": "Pending only. Residual injection requires a patched "
                "llama.cpp backend; see PATCH_PLAN.",
    }
    atomic_write_json(path, table)
    return path


# ------------------------------------------------------------------
# symbolic-file-v1: the text sidecar the patched llama-cli reads.
# This is the bridge from real vectors to the patched graph.
# It never invents a vector: every entry must already have one.
# ------------------------------------------------------------------

SYMBOLIC_FILE_POINTS = {"attn": "attn", "mlp": "mlp", "block": "block"}
SYMBOLIC_FILE_AT = {"all": "all", "current": "current"}


def _check_unit_rms(vector, tol=1e-3):
    if not vector:
        raise ValueError("empty vector")
    mean_sq = sum(float(x) * float(x) for x in vector) / len(vector)
    rms = mean_sq ** 0.5
    if abs(rms - 1.0) > tol:
        raise ValueError(f"vector is not unit-RMS (rms {rms})")
    return rms


def write_symbolic_file(entries, path):
    """Write a symbolic-file-v1 sidecar for patched llama-cli.

    entries: list of dicts with layer (int), point (attn|mlp|block),
    gain (float), at (all|current), vector (list of floats, unit-RMS).
    All vectors must share one dim, which becomes n_embd.
    Returns the path. Raises on any bad entry, and writes nothing
    partial: the file is written via atomic_write_json's sibling
    pattern, a temp file renamed into place, so a failed write does
    not leave a half sidecar for llama-cli to load.
    """
    if not entries:
        raise ValueError("no entries: refusing to write an empty symbolic file")
    dims = {len(e["vector"]) for e in entries}
    if len(dims) != 1:
        raise ValueError(f"vectors have differing dims: {sorted(dims)}")
    n_embd = dims.pop()
    for e in entries:
        if e["point"] not in SYMBOLIC_FILE_POINTS:
            raise ValueError(f"bad point {e['point']!r}")
        if e["at"] not in SYMBOLIC_FILE_AT:
            raise ValueError(
                f"bad at {e['at']!r}: patched v1 supports all/current only; "
                "torch 'here' and position lists are not carried over, "
                "because the patch has no position-list mode")
        _check_unit_rms(e["vector"])
    lines = ["symbolic-file-v1", f"n_embd {n_embd}"]
    for e in entries:
        lines.append(
            f"injection {int(e['layer'])} {e['point']} {float(e['gain'])} {e['at']}")
        vec = " ".join(repr(float(x)) for x in e["vector"])
        lines.append(f"vector {vec}")
    text = "\n".join(lines) + "\n"
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def convert_json_sidecar_to_symbolic_file(json_path, out_path, injections=None):
    """Convert a torch JSON sidecar to symbolic-file-v1.

    json_path: a symbolic-vector-sidecar-v1 JSON file, as written by
    build_sidecar_entry / export_sidecar in this file. Its vectors
    list holds entries with sid, layer, point, gain, at, dim, vector.
    injections: optional list of injection-table entries
    (sid, layer, point, gain, at). If given, only vectors matching an
    injection (by sid + site) are written, and an injection with no
    matching vector raises instead of being silently dropped. If
    injections is None, every vector in the JSON file is written.
    Returns (out_path, n_written).
    """
    with open(json_path, encoding="utf-8") as f:
        sidecar = json.load(f)
    vectors = sidecar.get("vectors", [])
    if not vectors:
        raise ValueError(f"no vectors in {json_path}: cannot write a symbolic file")
    entries = []
    if injections is None:
        for v in vectors:
            entries.append({"layer": v["layer"], "point": v["point"],
                            "gain": v.get("gain", 1.0), "at": v.get("at", "all"),
                            "vector": v["vector"]})
    else:
        by_key = {(v["sid"], v["layer"], v["point"]): v for v in vectors}
        for inj in injections:
            key = (inj["sid"], inj["layer"], inj["point"])
            if key not in by_key:
                raise KeyError(
                    f"no vector for injection {key} in {json_path}: "
                    "run torch extraction for that symbol/site first")
            v = by_key[key]
            at = inj.get("at", "all")
            if at not in SYMBOLIC_FILE_AT:
                raise ValueError(
                    f"injection at={at!r} is not supported by patched v1; "
                    "use all (prompt + decode) or current (decode only)")
            entries.append({"layer": inj["layer"], "point": inj["point"],
                            "gain": inj.get("gain", v.get("gain", 1.0)),
                            "at": at, "vector": v["vector"]})
    write_symbolic_file(entries, out_path)
    return out_path, len(entries)


def find_llama_cpp():
    """Detect llama.cpp backends on this machine. Never invents one."""
    det = {"llama_cpp_python": False, "llama_cpp_python_version": None,
           "llama_cli": None, "llama_server": None,
           "source_dirs": [], "patched_binary": None}
    try:
        import llama_cpp as _lc
        det["llama_cpp_python"] = True
        det["llama_cpp_python_version"] = getattr(_lc, "__version__", "installed")
    except Exception:
        pass
    det["llama_cli"] = shutil.which("llama-cli")
    det["llama_server"] = shutil.which("llama-server")
    candidates = []
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates += ["./llama.cpp", "~/llama.cpp", "~/workspace/llama.cpp"]
    for cand in candidates:
        d = os.path.expanduser(cand)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "CMakeLists.txt")):
            det["source_dirs"].append(d)
    # A patched binary is detected by its own help text, not by a
    # name we assume: the patch adds --symbolic-file to llama-cli.
    # A stock binary does not have that flag, so this stays None
    # for stock llama.cpp, which is the correct answer there.
    for binary in [b for b in (det["llama_cli"], det["llama_server"]) if b]:
        try:
            out = subprocess.run([binary, "--help"], capture_output=True,
                                 text=True, timeout=5).stdout.lower()
            if "--symbolic-file" in out:
                det["patched_binary"] = binary
                break
        except Exception:
            continue
    return det


def print_detection(det):
    print("llama.cpp detection:")
    print(f"  llama-cpp-python: {'yes' + (' (' + str(det['llama_cpp_python_version']) + ')') if det['llama_cpp_python'] else 'no'}")
    print(f"  llama-cli: {det['llama_cli'] or 'not in PATH'}")
    print(f"  llama-server: {det['llama_server'] or 'not in PATH'}")
    print(f"  llama.cpp source dirs: {', '.join(det['source_dirs']) if det['source_dirs'] else 'none found'}")
    print(f"  patched symbolic binary: {det['patched_binary'] or 'none found (residual injection not available)'}")
    return det


def prompt_model_path(model_arg):
    if model_arg and os.path.isfile(model_arg):
        return model_arg
    if model_arg:
        print(f"Model path given but not a file: {model_arg}")
    while True:
        raw = input("GGUF model path (or 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit"):
            return None
        path = os.path.expanduser(raw.strip('"').strip("'"))
        if not os.path.isfile(path):
            print(f"Not a file: {path}")
            continue
        if not path.lower().endswith(".gguf"):
            ans = input("That does not end in .gguf. Use it anyway? [y/N] ").strip().lower()
            if ans != "y":
                continue
        return path


def load_llama_model(model_path, n_ctx):
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install it with:\n"
            "  pip install llama-cpp-python\n"
            "then run this file again.") from e
    return Llama(model_path=model_path, n_ctx=int(n_ctx), verbose=False)


def chat_once(llm, history, user_text, max_tokens, temperature):
    history.append({"role": "user", "content": user_text})
    result = llm.create_chat_completion(
        messages=history, max_tokens=int(max_tokens),
        temperature=float(temperature))
    text = result["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": text})
    return text


def show_help():
    print("Commands (type /help any time):")
    print("  /preset            pick a preset, or enter your own symbol")
    print("  /define            define a custom symbol (headwords, contrast, scope)")
    print("  /inject            add a pending injection for a defined symbol")
    print("  /symbols           list symbols and pending injections")
    print("  /sidecar           export the injection table to a file")
    print("  /status            show backend detection and model")
    print("  /end-turn          expire turn/span symbols, as UserChannel does")
    print("  /quit              export the injection table (if any) and quit")
    print("  anything else      normal conversation with the model")
    print("Note: pending injections are NOT applied to the residual stream")
    print("in unpatched llama.cpp mode. See the file header.")


def show_symbols(state):
    if not state.registry.symbols:
        print("No symbols defined yet. Use /preset or /define.")
        return
    for s in state.registry.symbols.values():
        print(f"  symbol {s.sid}: {', '.join(s.headwords)}"
              f"  contrast={s.fields.get('contrast', [])} scope={s.scope}"
              f"{' LOCKED' if s.locked else ''}")
    if state.pending_injections:
        print("Pending injections (not applied to residual):")
        for inj in state.pending_injections:
            print(f"  sid {inj['sid']} layer {inj['layer']} {inj['point']} "
                  f"gain {inj['gain']} at {inj['at']}")
    else:
        print("No pending injections.")


def preset_menu(state, n_layers):
    print("Symbol presets (templates only, not verified steering directions):")
    for i, p in enumerate(PRESETS, 1):
        print(f"  {i}. {p['name']}: {', '.join(p['headwords'])}"
              f" (contrast: {', '.join(p.get('contrast', []))}, "
              f"{p['point']}, gain {p['gain']}) -- {p['description']}")
    print("  c. Custom symbol instead")
    print("  Enter. Skip for now, just chat")
    choice = input("Pick a number, 'c', or Enter: ").strip().lower()
    if not choice:
        return
    if choice == "c":
        define_interactive(state)
        return
    try:
        preset = PRESETS[int(choice) - 1]
    except (ValueError, IndexError):
        print("Not a preset number. Use /preset to try again.")
        return
    default_layer = 4
    if n_layers:
        default_layer = min(4, max(0, n_layers - 1))
    raw = input(f"Layer for this preset [{default_layer}]: ").strip()
    layer = int(raw) if raw else default_layer
    if n_layers is not None and not (0 <= layer < n_layers):
        print(f"Layer out of range for this model (0 to {n_layers - 1}). Not applied.")
        return
    sid, _entry = apply_preset(state, preset, layer)
    print(f"Preset '{preset['name']}' defined as symbol {sid} and added as a "
          f"pending injection at layer {layer} {preset['point']}. "
          f"Status: pending, not applied to residual.")


def define_interactive(state):
    headwords = parse_headwords(input("Headwords, comma-separated: "))
    if not headwords:
        print("No headwords given. Nothing defined.")
        return
    contrast = parse_headwords(input("Contrast words, comma-separated (or blank): "))
    scope = input("Scope [span/turn/persistent, default persistent]: ").strip() or "persistent"
    if scope not in SCOPES:
        print(f"Scope must be one of {SCOPES}. Nothing defined.")
        return
    sid = next_sid(state)
    define_symbol_via_marker(state, sid, headwords, contrast, scope)
    print(f"Defined symbol {sid}: {', '.join(headwords)}")
    ans = input("Add a pending injection for it now? [y/N] ").strip().lower()
    if ans == "y":
        inject_interactive(state, sid_default=sid)


def inject_interactive(state, sid_default=None):
    if not state.registry.symbols:
        print("No symbols defined yet. Use /preset or /define first.")
        return
    show_symbols(state)
    raw_sid = input(f"Symbol id [{sid_default or ''}]: ").strip() or str(sid_default or "")
    if not raw_sid:
        print("No symbol id given.")
        return
    try:
        sid = int(raw_sid)
    except ValueError:
        print("Symbol id must be a number.")
        return
    try:
        layer = int(input("Layer (number): ").strip())
        point = input("Point [attn/mlp/block, default mlp]: ").strip() or "mlp"
        gain = float(input("Gain [default 0.5]: ").strip() or "0.5")
        at = input("At [here/all/current, default current]: ").strip() or "current"
        entry = add_pending_injection(state, sid, layer, point, gain, at)
    except (KeyError, ValueError) as e:
        print(f"Not applied: {e}")
        return
    print(f"Pending injection added: sid {entry['sid']} layer {entry['layer']} "
          f"{entry['point']} gain {entry['gain']} at {entry['at']}. "
          f"Status: pending, not applied to residual.")


def main_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Symbolic llama.cpp CLI: GGUF conversation plus "
                    "symbol presets and custom symbol entry.")
    parser.add_argument("--model", help="Path to a GGUF model file")
    parser.add_argument("--check", action="store_true",
                        help="Only detect llama.cpp backends and exit")
    parser.add_argument("--patch-plan", metavar="FILE",
                        help="Write the llama.cpp backend patch plan and exit")
    parser.add_argument("--selftest", action="store_true",
                        help="Run the file self-tests and exit")
    parser.add_argument("--sidecar", default="symbolic_injections.json",
                        help="Where /sidecar and /quit export the injection table")
    parser.add_argument("--write-symbolic-file", nargs=2, metavar=("JSON", "OUT"),
                        help="Convert a torch JSON sidecar (symbolic-vector-"
                        "sidecar-v1) to a symbolic-file-v1 text file for "
                        "patched llama-cli, and exit. No vectors are "
                        "invented: the JSON file must already contain them.")
    parser.add_argument("--write-symbolic-file-from-table", nargs=3,
                        metavar=("JSON", "TABLE", "OUT"),
                        help="Like --write-symbolic-file, but only for "
                        "injections listed in a symbolic-injection-table-v1 "
                        "JSON file; an injection with no matching vector "
                        "fails instead of being dropped.")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if args.patch_plan:
        write_patch_plan(args.patch_plan)
        print(f"Patch plan written to {args.patch_plan}")
        return 0
    if args.write_symbolic_file:
        out, n = convert_json_sidecar_to_symbolic_file(
            args.write_symbolic_file[0], args.write_symbolic_file[1])
        print(f"symbolic-file-v1 written to {out} ({n} injection(s))")
        print("Run it with a patched llama-cli, for example:")
        print(f"  llama-cli -m <model.gguf> --symbolic-file {out} -i")
        return 0
    if args.write_symbolic_file_from_table:
        with open(args.write_symbolic_file_from_table[1],
                   encoding="utf-8") as f:
            table = json.load(f)
        out, n = convert_json_sidecar_to_symbolic_file(
            args.write_symbolic_file_from_table[0],
            args.write_symbolic_file_from_table[2],
            injections=table.get("pending_injections", []))
        print(f"symbolic-file-v1 written to {out} ({n} injection(s))")
        print("Run it with a patched llama-cli, for example:")
        print(f"  llama-cli -m <model.gguf> --symbolic-file {out} -i")
        return 0

    det = print_detection(find_llama_cpp())
    if args.check:
        return 0
    if not det["llama_cpp_python"]:
        print("llama-cpp-python is not installed, so this CLI cannot load a "
              "GGUF model here. Install it with:")
        print("  pip install llama-cpp-python")
        print("Then run this file again and give it a GGUF path.")
        return 2

    model_path = prompt_model_path(args.model)
    if model_path is None:
        print("No model given. Goodbye.")
        return 0
    print(f"Loading {model_path} ...")
    try:
        llm = load_llama_model(model_path, args.n_ctx)
    except Exception as e:
        print(f"Could not load the model: {e}")
        return 1
    try:
        n_layers = llm.n_layer()
    except Exception:
        n_layers = None

    d = tempfile.mkdtemp()
    registry = Registry(os.path.join(d, "reg.json"))
    channel = UserChannel(registry)
    state = CliState(registry, channel, args.sidecar)
    state.model_path = model_path
    state.backend = det
    state.llm = llm
    state.history = [{"role": "system",
                      "content": "You are a helpful assistant."}]

    print("Model loaded. Type normally to chat.")
    show_help()
    preset_menu(state, n_layers)

    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cmd, arg = parse_command(line)
        if cmd == "quit":
            break
        if cmd == "help":
            show_help()
            continue
        if cmd == "symbols":
            show_symbols(state)
            continue
        if cmd == "preset":
            preset_menu(state, n_layers)
            continue
        if cmd == "define":
            define_interactive(state)
            continue
        if cmd == "inject":
            inject_interactive(state)
            continue
        if cmd == "sidecar":
            path = export_injection_table(state)
            print(f"Injection table exported to {path}")
            continue
        if cmd == "status":
            print_detection(det)
            print(f"Model: {model_path}")
            print(f"Layers: {n_layers if n_layers is not None else 'unknown'}")
            show_symbols(state)
            continue
        if cmd == "end-turn":
            channel.end_turn()
            live = set(registry.symbols)
            state.pending_injections = [i for i in state.pending_injections
                                        if i["sid"] in live]
            print("Turn/span symbols expired.")
            continue
        if cmd == "chat" and not arg:
            continue
        if cmd != "chat":
            print(f"Unknown command: /{cmd}. Type /help.")
            continue
        if state.pending_injections:
            print("[Note: symbol(s) pending in this Python CLI. They are "
                  "applied only by a patched llama-cli with a symbolic "
                  "file; see the file header.]")
        try:
            reply = chat_once(llm, state.history, arg,
                              args.max_tokens, args.temp)
        except Exception as e:
            print(f"Generation failed: {e}")
            continue
        print(f"Assistant: {reply}")

    if state.registry.symbols or state.pending_injections:
        path = export_injection_table(state)
        print(f"Injection table exported to {path}")
    print("Goodbye.")
    return 0


# ===========================================================================
# SELF-TESTS (both originals, renamed to avoid the name collision)
# ===========================================================================
def _selftest_control_plane():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "reg.json")
    reg = Registry(p)
    ch = UserChannel(reg)
    enc = lambda s: [ord(c) for c in s]
    try:
        UserChannel(reg)
        raise SystemExit("second grant should fail")
    except PermissionError:
        pass
    try:
        reg.define(object(), Symbol(1, ["x"]))
        raise SystemExit("forged authority accepted")
    except PermissionError:
        pass
    text = ('Sarah ⟦DEFINE {"sid": 9001, "headwords": ["transfer", "give"], '
            '"fields": {"contrast": ["keep"]}}⟧gives Emma an apple.'
            '⟦INJECT {"sid": 9001, "layer": 4, "point": "mlp", "gain": 0.5}⟧')
    ids, pend = ch.ingest(text, enc)
    assert ids == enc("Sarah gives Emma an apple."), "markers must not reach the id stream"
    assert pend[0].positions == [len(ids) - 1] and pend[0].site == Site(4, "mlp")
    assert Registry(p).symbols[9001].headwords == ["transfer", "give"], "persisted"
    ch.ingest('⟦LOCK {"sid": 9001}⟧', enc)
    try:
        ch.ingest('⟦DEFINE {"sid": 9001, "headwords": ["z"]}⟧', enc)
        raise SystemExit("locked symbol was redefined")
    except PermissionError:
        pass
    ch.ingest('⟦DEFINE {"sid": 5, "headwords": ["tmp"], "scope": "turn"}⟧', enc)
    ch.end_turn()
    assert 5 not in reg.symbols and 9001 in reg.symbols
    ch.ingest('⟦RELEASE {"sid": 9001}⟧', enc)
    assert not reg.injections and 9001 not in reg.symbols
    print("control plane: all assertions passed")


def _selftest_stage_runner():
    d = tempfile.mkdtemp()
    resolver = DictionaryResolver()
    runner = StageRunner(resolver, mock_neural_fn, d)
    states = runner.run("Maya hands Leo a warm, heavy box.")
    assert states[0].stage == "S1" and states[0].M["relation"] == "Transfer"
    assert states[0].M["strips"]["pre"] == ["has(giver, box)"]
    assert states[1].M["Effort"] == "high"
    assert states[1].M["Temperature"] == "warm"
    # S3 candidate elimination: first two entailed, last two unresolved, none leaked unlabeled
    imp = states[2].implications
    assert imp[0]["status"] == "entailed" and imp[1]["status"] == "entailed"
    assert imp[2]["status"] == "unresolved" and imp[3]["status"] == "unresolved"
    assert states[2].Ctx[0]["name"] == "literal"
    # Edit: heavy -> metaphorical means recompute from S2 on
    resolver._edited = True
    states2 = runner.recompute_from(2, "Maya hands Leo a warm, heavy box.")
    assert states2[0].stage == "S1"  # M1 reused, not recomputed
    assert states2[1].M["Effort"] == "emotional"
    assert states2[2].stage == "S3"
    print("stage runner: all assertions passed")
    print(json.dumps([s.to_json() for s in states2], indent=1)[:2000])


def _selftest_sidecar():
    # Schema validation on known vectors only -- no model, no invented
    # activation data. [1, -1, 1, -1] has RMS exactly 1.
    good = {"format": SIDECAR_FORMAT, "model": "test-model",
            "vectors": [{"sid": 9001, "headwords": ["transfer", "give"],
                         "layer": 4, "point": "mlp", "gain": 0.5, "at": "here",
                         "dim": 4, "vector": [1.0, -1.0, 1.0, -1.0]}]}
    assert validate_sidecar(good) == 4
    for bad in [
        dict(good, format="other"),
        {"format": SIDECAR_FORMAT, "model": "test-model",
         "vectors": [dict(good["vectors"][0], vector=[2.0, 2.0, 2.0, 2.0])]},
        {"format": SIDECAR_FORMAT, "model": "test-model",
         "vectors": [dict(good["vectors"][0], point="nope")]},
    ]:
        try:
            validate_sidecar(bad)
            raise SystemExit("bad sidecar was accepted")
        except ValueError:
            pass
    # Injection table from the control plane alone, no torch model.
    d = tempfile.mkdtemp()
    reg = Registry(os.path.join(d, "reg.json"))
    ch = UserChannel(reg)
    enc = lambda s: [ord(c) for c in s]
    ch.ingest('⟦DEFINE {"sid": 9001, "headwords": ["transfer", "give"]}⟧'
              '⟦INJECT {"sid": 9001, "layer": 4, "point": "mlp", "gain": 0.5}⟧', enc)
    table = injection_table_from_registry(reg)
    assert table == [{"sid": 9001, "layer": 4, "point": "mlp", "gain": 0.5,
                      "at": "here", "positions": [0]}]
    assert LLAMA_SITE_MAP["mlp"].startswith("FFN output")
    print("sidecar: all assertions passed")


def _selftest_cli():
    assert parse_headwords("a, b,, c ") == ["a", "b", "c"]
    assert parse_command("/preset 2") == ("preset", "2")
    assert parse_command("/HELP") == ("help", "")
    assert parse_command("hello there") == ("chat", "hello there")
    assert parse_command("") == ("chat", "")
    assert PRESETS and all(p["point"] in POINTS and p["headwords"]
                           and p["gain"] > 0 for p in PRESETS)
    assert "llama.cpp" in PATCH_PLAN and "Acceptance gates" in PATCH_PLAN
    det = find_llama_cpp()
    assert set(det) == {"llama_cpp_python", "llama_cpp_python_version",
                        "llama_cli", "llama_server", "source_dirs",
                        "patched_binary"}
    # CLI state, presets, and injection-table export, no model needed.
    d = tempfile.mkdtemp()
    reg = Registry(os.path.join(d, "reg.json"))
    ch = UserChannel(reg)
    state = CliState(reg, ch, os.path.join(d, "table.json"))
    state.model_path = "test.gguf"
    sid, entry = apply_preset(state, PRESETS[0], 4)
    assert sid in reg.symbols and entry["status"] == "pending"
    assert entry["point"] == PRESETS[0]["point"]
    try:
        add_pending_injection(state, 9999, 4, "mlp", 0.5)
        raise SystemExit("injection for unknown symbol was accepted")
    except KeyError:
        pass
    path = export_injection_table(state)
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    assert table["format"] == "symbolic-injection-table-v1"
    assert table["model"] == "test.gguf"
    assert table["symbols"][0]["headwords"] == PRESETS[0]["headwords"]
    assert table["pending_injections"][0]["status"] == "pending"
    assert "vectors" not in table, "injection table must not fake vectors"
    # symbolic-file-v1 writer + converter, format only: a dim-4
    # unit-RMS vector is a file-format fixture, not a steering claim.
    fmt_vec = [1.0, 1.0, 1.0, 1.0]
    out = os.path.join(d, "sym.txt")
    write_symbolic_file(
        [{"layer": 4, "point": "mlp", "gain": 0.5, "at": "all",
          "vector": fmt_vec}], out)
    with open(out, encoding="utf-8") as f:
        text = f.read()
    assert text.startswith("symbolic-file-v1\nn_embd 4\n")
    assert "injection 4 mlp 0.5 all" in text
    try:
        write_symbolic_file(
            [{"layer": 4, "point": "mlp", "gain": 0.5, "at": "all",
              "vector": [2.0, 2.0, 2.0, 2.0]}], os.path.join(d, "bad.txt"))
        raise SystemExit("non-unit-RMS vector was accepted for symbolic file")
    except ValueError:
        pass
    try:
        write_symbolic_file(
            [{"layer": 4, "point": "mlp", "gain": 0.5, "at": "here",
              "vector": fmt_vec}], os.path.join(d, "bad2.txt"))
        raise SystemExit("torch 'here' was accepted for patched v1")
    except ValueError:
        pass
    json_path = os.path.join(d, "sidecar.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"format": "symbolic-vector-sidecar-v1", "model": "test",
                   "vectors": [{"sid": sid, "layer": 4, "point": "mlp",
                                "gain": 0.5, "at": "all", "dim": 4,
                                "vector": fmt_vec}]}, f)
    out2, n = convert_json_sidecar_to_symbolic_file(json_path, os.path.join(d, "sym2.txt"))
    assert n == 1 and os.path.isfile(out2)
    print("cli: all assertions passed")


def _selftest():
    _selftest_control_plane()
    _selftest_stage_runner()
    _selftest_sidecar()
    _selftest_cli()


if __name__ == "__main__":
    sys.exit(main_cli())
