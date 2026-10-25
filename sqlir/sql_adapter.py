"""Compiled SQL-derived model exposed through the user ModelAdapter interface."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .conical_sql_ir_transformer import UNRESOLVED, encode_cell, norm, relu, softmax_causal
from .lm_compile import SPECIAL_WORDS, norm_word
from .model_adapter import LayerInfo, ModelAdapter
from .runtime import CompiledModel
from .weights import DirectWeights, RubikWeights, WeightSource


READOUT = {"bigram": 1.0, "colloc": 0.5, "unigram": 0.05}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")
DEFINE_RE = re.compile(
    r"^\s*(?:define|definition of|meaning of|what does)\s+(\S+?)(?:\s+mean)?\s*\??\s*$",
    re.IGNORECASE,
)
WHAT_IS_RE = re.compile(
    r"^\s*what\s+is\s+(?:an?\s+|the\s+)?(\S+?)\s*\??\s*$", re.IGNORECASE
)
RELATED_RE = re.compile(
    r"^\s*(?:related to|synonyms? of|relations? of)\s+(\S+?)\s*\??\s*$",
    re.IGNORECASE,
)
DISCLOSURE = (
    "count-based next-word model over the compiled DB behind the ModelAdapter "
    "interface; outputs only attested word transitions; not a trained LLM; no reasoning"
)


class SQLAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str = "model",
        verbose: bool = False,
        seed: int = 0,
        weights: WeightSource | None = None,
        weight_mode: str = "direct",
        rubik_store: str | Path | None = None,
        rubik_module: str | Path | None = None,
    ):
        super().__init__(model_name, verbose)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._model: CompiledModel | None = None
        self._weights = weights
        self._weight_mode = weight_mode
        self._rubik_store = rubik_store
        self._rubik_module = rubik_module
        self._vocab_offsets = None
        self._vocab_keys = None
        self._last_surface: list[str] = []
        self._state: dict[int, Any] = {}
        self._residual: dict[int, Any] = {}
        self._pending_layer_states: dict[int, dict[str, Any]] = {}
        self._attention = None
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        self._pending_kv = None
        self._pending_logits = None
        self._logits = None
        self._workbook = None

    @property
    def model(self) -> CompiledModel:
        if self._model is None:
            self.load()
        return self._model

    @property
    def lm(self) -> dict:
        return self.model.manifest.get("lm", {})

    def load(self) -> None:
        if self.is_loaded:
            return
        self._model = CompiledModel(self.model_name, verify="lazy")
        self._vocab_offsets = self.model._segment_array(self.lm["segments"]["vocab"]["offsets"])
        self._vocab_keys = self.model._segment_bytes(self.lm["segments"]["vocab"]["keys"])
        if self._weights is None:
            if self._weight_mode == "direct":
                self._weights = DirectWeights(self.model)
            elif self._weight_mode == "rubik":
                if self._rubik_store is None or self._rubik_module is None:
                    raise ValueError("rubik mode requires --rubik-store and --rubik-module")
                self._weights = RubikWeights(self._rubik_store, self._rubik_module)
            else:
                raise ValueError(f"Unknown weight mode: {self._weight_mode}")
        self.is_loaded = True

    def unload(self) -> None:
        self._model = None
        self._weights = None
        self.is_loaded = False

    def _id_for_word(self, word: str) -> int:
        self.load()
        if word in SPECIAL_WORDS:
            return SPECIAL_WORDS.index(word)
        try:
            word = norm_word(word)
        except ValueError:
            return 0
        encoded = word.encode("utf-8")
        offsets = self._vocab_offsets
        keys = self._vocab_keys
        low, high = 3, len(offsets) - 1
        while low < high:
            middle = (low + high) // 2
            candidate = bytes(keys[int(offsets[middle]) : int(offsets[middle + 1])])
            if candidate < encoded:
                low = middle + 1
            else:
                high = middle
        if low >= len(offsets) - 1:
            return 0
        candidate = bytes(keys[int(offsets[low]) : int(offsets[low + 1])])
        return low if candidate == encoded else 0

    def _word_for_id(self, token_id: int) -> str:
        if token_id < 0 or token_id >= len(self._vocab_offsets) - 1:
            return SPECIAL_WORDS[0]
        start = int(self._vocab_offsets[token_id])
        end = int(self._vocab_offsets[token_id + 1])
        return bytes(self._vocab_keys[start:end]).decode("utf-8")

    def tokenize(self, text: str) -> List[int]:
        self.load()
        self._last_surface = TOKEN_RE.findall(text)
        result = []
        for surface in self._last_surface:
            try:
                word = norm_word(surface)
            except ValueError:
                word = SPECIAL_WORDS[0]
            result.append(self._id_for_word(word))
        return result

    def detokenize(self, tokens: List[int]) -> str:
        pieces = [self._word_for_id(int(token)) for token in tokens]
        pieces = [piece for piece in pieces if piece not in SPECIAL_WORDS]
        output = ""
        for piece in pieces:
            if not output:
                output = piece
            elif piece in ".,;:!?)'\"":
                output += piece
            elif output.endswith("("):
                output += piece
            else:
                output += " " + piece
        return output

    def _channel(self, name: str) -> np.ndarray:
        self.load()
        return self._weights.get(name)

    def _csr_row(self, group: str, token_id: int) -> tuple[np.ndarray, np.ndarray]:
        segments = self.lm["segments"][group]
        indptr = self.model._segment_array(segments["indptr"])
        start, end = int(indptr[token_id]), int(indptr[token_id + 1])
        cols = self.model._segment_array(segments["col"])[start:end]
        if "value" in segments:
            values = self.model._segment_array(segments["value"])[start:end]
        else:
            values = np.arange(start, end, dtype=np.int64)
        return cols, values

    def _text_row(
        self, group: str, token_id: int, limit: int | None = None
    ) -> list[Any]:
        segments = self.lm["segments"][group]
        indptr = self.model._segment_array(segments["indptr"])
        start, end = int(indptr[token_id]), int(indptr[token_id + 1])
        if limit is not None:
            end = min(end, start + limit)
        offsets = self.model._segment_array(segments["offsets"])
        raw = self.model._segment_bytes(segments["rows"])
        return [
            json.loads(bytes(raw[int(offsets[index]) : int(offsets[index + 1])]))
            for index in range(start, end)
        ]

    def _text_row_count(self, group: str, token_id: int) -> int:
        indptr = self.model._segment_array(self.lm["segments"][group]["indptr"])
        return int(indptr[token_id + 1]) - int(indptr[token_id])

    def _embedding(self, token_id: int) -> np.ndarray:
        word = self._word_for_id(token_id)
        if token_id < 3:
            return np.zeros(437, dtype=np.float64)
        E = self._channel("channel:E")
        vectors = [E[ord(char) - 32] for char in word if 32 <= ord(char) <= 126]
        return np.mean(vectors, axis=0) if vectors else np.zeros(437, dtype=np.float64)

    @staticmethod
    def _apply_hook(callback, state):
        try:
            result = callback(state)
        except TypeError:
            result = callback(0, state)
        return state if result is None else np.asarray(result, dtype=np.float64)

    def forward(
        self, tokens: List[int], cache: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        self.load()
        if not tokens:
            raise ValueError("tokens must not be empty")
        token_ids = [int(token) for token in tokens]
        X = np.stack([self._embedding(token) for token in token_ids]).astype(np.float64)
        pending = self._residual.pop(0, None)
        if pending is not None:
            pending = np.asarray(pending, dtype=np.float64)
            X = X + pending if pending.ndim == 2 else X + pending[None, :]
        hook = getattr(self, "_active_hooks", {}).get(0)
        if hook is not None:
            X = self._apply_hook(hook, X)
        Z1 = norm(X)
        Q = Z1 @ self._channel("channel:B:Q")
        K = Z1 @ self._channel("channel:B:K")
        V = Z1 @ self._channel("channel:B:V")
        layer_zero_writes = self._pending_layer_states.pop(0, {})
        override = self._pending_kv
        self._pending_kv = None
        if override is not None and 0 in override:
            K = np.asarray(override[0]["key"], dtype=np.float64)
            V = np.asarray(override[0]["value"], dtype=np.float64)
        A = softmax_causal((Q @ K.T) / np.sqrt(437))
        if "attention" in layer_zero_writes:
            A = np.asarray(layer_zero_writes["attention"], dtype=np.float64)
            if A.shape != (len(token_ids), len(token_ids)):
                raise ValueError("attention state has the wrong shape")
        O = (A @ V) @ self._channel("channel:B:O")
        Z2 = X + O
        Z3 = norm(Z2)
        U = relu(Z3 @ self._channel("channel:B:F1"))
        if "mlp" in layer_zero_writes:
            U = np.asarray(layer_zero_writes["mlp"], dtype=np.float64)
            if U.shape != X.shape:
                raise ValueError("MLP state has the wrong shape")
        Y = Z2 + U @ self._channel("channel:B:F2")
        pending_layer_one = self._residual.pop(1, None)
        if pending_layer_one is not None:
            pending_layer_one = np.asarray(pending_layer_one, dtype=np.float64)
            Y = Y + (
                pending_layer_one
                if pending_layer_one.ndim == 2
                else pending_layer_one[None, :]
            )
        if Y.shape != X.shape or not np.isfinite(Y).all():
            raise ValueError("Forward pass produced invalid output")
        self._state = {
            0: {
                "Q": Q.copy(),
                "K": K.copy(),
                "V": V.copy(),
                "A": A.copy(),
                "U": U.copy(),
                "Z2": Z2.copy(),
            },
            1: Y.copy(),
        }
        self._attention = A.copy()
        self._cache = deepcopy(cache or {})
        self._cache[0] = {"key": K.copy(), "value": V.copy()}
        logits = self._readout(token_ids, A)
        if self._pending_logits is not None:
            logits = logits + np.asarray(self._pending_logits, dtype=np.float64)
            self._pending_logits = None
        self._logits = logits.copy()
        return logits.copy(), deepcopy(self._cache)

    def _readout(self, tokens: list[int], attention: np.ndarray) -> np.ndarray:
        vocab_size = self.get_vocab_size()
        last = tokens[-1]
        bigram = np.zeros(vocab_size, dtype=np.float64)
        columns, values = self._csr_row("bigram", last)
        bigram[columns.astype(np.int64)] = values
        colloc = np.zeros(vocab_size, dtype=np.float64)
        for token, weight in zip(tokens, attention[-1]):
            if weight:
                cols, vals = self._csr_row("colloc", token)
                colloc[cols.astype(np.int64)] += float(weight) * vals
        unigram = self.model._segment_array(self.lm["segments"]["unigram"])
        logits = np.full(vocab_size, -np.inf, dtype=np.float64)
        allowed = (bigram != 0) | (colloc != 0)
        if np.any(allowed):
            logits[allowed] = (
                READOUT["bigram"] * np.log1p(bigram[allowed])
                + READOUT["colloc"] * np.log1p(colloc[allowed])
                + READOUT["unigram"] * np.log1p(unigram[allowed])
            )
            logits[2] = (
                READOUT["bigram"] * np.log1p(bigram[2])
                if bigram[2] > 0
                else -np.inf
            )
        else:
            logits[2] = 0.0
        logits[0] = logits[1] = -np.inf
        return logits

    def get_layer_info(self, layer_id: int) -> LayerInfo:
        if layer_id not in (0, 1):
            raise ValueError("layer_id must be 0 or 1")
        return LayerInfo(layer_id, 1, 437, 437, f"sqlir-{layer_id}")

    def read_layer_state(self, layer_id: int, layer_type: str = "residual") -> np.ndarray:
        state = self._state.get(layer_id)
        if state is None:
            raise RuntimeError("No forward state available")
        if layer_type == "residual":
            return deepcopy(state)
        if layer_id == 0:
            key = {
                "attention": "A",
                "q": "Q",
                "k": "K",
                "v": "V",
                "mlp": "U",
                "z2": "Z2",
            }.get(layer_type, layer_type)
            if key in state:
                return np.array(state[key], copy=True)
        if layer_type == "kv":
            return self.read_kv_cache(layer_id)
        raise ValueError(f"Unknown layer state: {layer_type}")

    def write_layer_state(
        self, layer_id: int, state: np.ndarray, layer_type: str = "residual"
    ) -> None:
        if layer_type == "residual":
            self.write_residual_stream(layer_id, state)
            return
        if layer_id != 0:
            raise ValueError("only layer 0 supports non-residual state writes")
        if layer_type == "kv":
            if not isinstance(state, dict):
                raise ValueError("KV state must be a cache dictionary")
            self.write_kv_cache(state)
            return
        if layer_type not in {"attention", "mlp"}:
            raise ValueError(f"Unknown layer state: {layer_type}")
        self._pending_layer_states.setdefault(layer_id, {})[layer_type] = np.array(
            state, dtype=np.float64, copy=True
        )

    def read_kv_cache(self, layer_id: Optional[int] = None) -> Dict:
        if layer_id is None:
            return deepcopy(self._cache)
        return deepcopy({layer_id: self._cache[layer_id]})

    def write_kv_cache(self, kv_data: Dict) -> None:
        self._pending_kv = deepcopy(kv_data)

    def read_residual_stream(self, layer_id: int) -> np.ndarray:
        state = self._state.get(layer_id)
        if state is None:
            raise RuntimeError("No forward state available")
        value = state["Z2"] if layer_id == 0 and isinstance(state, dict) else state
        return np.array(value, copy=True)

    def write_residual_stream(self, layer_id: int, activation: np.ndarray) -> None:
        if layer_id not in (0, 1):
            raise ValueError("layer_id must be 0 or 1")
        self._residual[layer_id] = np.array(activation, dtype=np.float64, copy=True)

    def read_attention_weights(
        self, layer_id: int, head_id: Optional[int] = None
    ) -> np.ndarray:
        if layer_id != 0 or self._attention is None:
            raise ValueError("only layer 0 attention is available")
        return np.array(self._attention, copy=True)

    def read_logits(self) -> np.ndarray:
        if self._logits is None:
            raise RuntimeError("No forward logits available")
        return np.array(self._logits, copy=True)

    def modify_logits(self, modification: np.ndarray) -> None:
        modification = np.asarray(modification, dtype=np.float64)
        if modification.shape != (self.get_vocab_size(),):
            raise ValueError("logit modification has the wrong shape")
        self._pending_logits = modification.copy()

    def sample(
        self, logits: np.ndarray, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9
    ) -> int:
        values = np.asarray(logits, dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            raise ValueError("no finite logits")
        if temperature <= 0:
            best = np.max(values[finite])
            return int(finite[np.flatnonzero(values[finite] == best)[0]])
        order = sorted(finite.tolist(), key=lambda index: (-values[index], index))
        if top_k and top_k > 0:
            order = order[:top_k]
        scaled = np.asarray([values[index] / temperature for index in order])
        probabilities = np.exp(scaled - np.max(scaled))
        probabilities /= probabilities.sum()
        if 0 < top_p < 1:
            keep = np.cumsum(probabilities) <= top_p
            keep[0] = True
            if keep.any() and not keep[-1]:
                first_false = int(np.flatnonzero(~keep)[0])
                keep[first_false] = True
            order = [index for index, include in zip(order, keep) if include]
            probabilities = probabilities[: len(order)]
            probabilities /= probabilities.sum()
        return int(self._rng.choice(order, p=probabilities))

    def get_vocab_size(self) -> int:
        return int(self.lm["vocab_size"])

    def get_num_layers(self) -> int:
        return 2

    def get_hidden_dim(self) -> int:
        return 437

    def _definition_values(self, token_id: int) -> list[str]:
        return [str(value) for value in self._text_row("definitions", token_id)]

    def _relation_values(self, token_id: int) -> list[list[Any]]:
        return [list(value) for value in self._text_row("relations", token_id)]

    def _procedure_word(self, prompt: str) -> str | None:
        match = DEFINE_RE.match(prompt) or WHAT_IS_RE.match(prompt)
        if match:
            return match.group(1)
        tokens = TOKEN_RE.findall(prompt)
        return tokens[0] if len(tokens) == 1 else None

    def _related_word(self, prompt: str) -> str | None:
        match = RELATED_RE.match(prompt)
        return match.group(1) if match else None

    def _source_label(self) -> str:
        source = self.lm.get("definition_source")
        return f"{source['table']}.{source['column']}" if source else "this database"

    def _lexicon_check(self, tokens: list[int], prompt_tokens: list[int] | None = None) -> dict:
        words = [
            {"word": self._word_for_id(token), "in_vocab": int(token) != 0}
            for token in tokens
        ]
        transitions = []
        if prompt_tokens:
            combined = prompt_tokens + tokens
            start = max(0, len(prompt_tokens) - 1)
            for left, right in zip(combined[start:], combined[start + 1 :]):
                count = self._transition_count(left, right)
                transitions.append({"from": self._word_for_id(left), "to": self._word_for_id(right), **count})
        return {"words": words, "transitions": transitions, "passed": all(
            item.get("bigram", 0) > 0 or item.get("colloc", 0) > 0
            for item in transitions
        )}

    def _transition_count(self, left: int, right: int) -> dict:
        bigram = 0
        cols, vals = self._csr_row("bigram", left)
        match = np.flatnonzero(cols == right)
        if len(match):
            bigram = int(vals[match[0]])
        colloc = 0.0
        cols, vals = self._csr_row("colloc", left)
        match = np.flatnonzero(cols == right)
        if len(match):
            colloc = float(vals[match[0]])
        return {"bigram": bigram, "colloc": colloc, "attested": bigram > 0 or colloc > 0}

    def _attested_mask(self, token_id: int) -> np.ndarray:
        mask = np.zeros(self.get_vocab_size(), dtype=bool)
        cols, values = self._csr_row("bigram", token_id)
        mask[cols.astype(np.int64)] = values > 0
        cols, values = self._csr_row("colloc", token_id)
        mask[cols.astype(np.int64)] |= values > 0
        return mask

    def _continuation(self, prompt: str, max_tokens: int, temperature: float, hooks=None):
        context = self.tokenize(prompt)
        generated: list[int] = []
        if not context:
            return generated, context
        self._active_hooks = hooks or {}
        try:
            for _ in range(max_tokens):
                logits, _ = self.forward(context + generated)
                if not np.isfinite(logits).any():
                    break
                supported = logits.copy()
                supported[~self._attested_mask((context + generated)[-1])] = -np.inf
                if not np.isfinite(supported).any():
                    break
                token = self.sample(supported, temperature=temperature)
                if token == 2:
                    break
                generated.append(token)
        finally:
            self._active_hooks = {}
        return generated, context

    def scan(self, text: str) -> dict:
        tokens = self.tokenize(text)
        unique = []
        seen = set()
        for token, surface in zip(tokens, self._last_surface):
            try:
                word = norm_word(surface)
            except ValueError:
                word = surface.strip().lower()
            if word not in seen:
                seen.add(word)
                unique.append(word)
        unknown = [word for word in unique if self._id_for_word(word) == 0]
        unique = unique[:64]
        workbook = self._workbook_data()
        entries = []
        for word in unique:
            token_id = self._id_for_word(word)
            in_vocab = token_id != 0
            item = {
                "word": word,
                "in_vocab": in_vocab,
                "token_id": token_id,
                "unigram": 0,
                "unigram_rank": 0,
                "pos": [],
                "lemma_of": [],
                "inflections": [],
                "definitions": [],
                "definitions_total": 0,
                "relations": [],
                "relations_total": 0,
                "next_by_bigram": [],
                "next_by_colloc": [],
                "workbook": self._workbook_for_word(word, workbook),
            }
            if in_vocab:
                unigram = self.model._segment_array(self.lm["segments"]["unigram"])
                ranks = self.model._segment_array(self.lm["segments"]["unigram_rank"])
                item["unigram"] = int(unigram[token_id])
                item["unigram_rank"] = int(ranks[token_id])
                item["pos"] = self._text_row("pos", token_id)
                item["lemma_of"] = self._text_row("lemma_of", token_id, limit=10)
                item["inflections"] = self._text_row("inflections", token_id, limit=10)
                item["definitions"] = self._text_row(
                    "definitions", token_id, limit=3
                )
                item["definitions_total"] = self._text_row_count(
                    "definitions", token_id
                )
                item["relations"] = self._text_row(
                    "relations", token_id, limit=10
                )
                item["relations_total"] = self._text_row_count(
                    "relations", token_id
                )
                cols, vals = self._csr_row("bigram", token_id)
                item["next_by_bigram"] = [
                    [self._word_for_id(int(column)), int(value)]
                    for column, value in sorted(zip(cols, vals), key=lambda pair: (-int(pair[1]), int(pair[0])))[:5]
                ]
                cols, vals = self._csr_row("colloc", token_id)
                item["next_by_colloc"] = [
                    [self._word_for_id(int(column)), float(value)]
                    for column, value in sorted(zip(cols, vals), key=lambda pair: (-float(pair[1]), int(pair[0])))[:5]
                ]
            entries.append(item)
        pairs = []
        for left, right in zip(tokens, tokens[1:]):
            transition = self._transition_count(left, right)
            pairs.append({
                "from": self._word_for_id(left),
                "to": self._word_for_id(right),
                **transition,
            })
        focus = None
        if tokens:
            self.forward(tokens)
            focus = self._word_for_id(tokens[int(np.argmax(self._attention[-1]))])
        return {
            "words": entries,
            "pairs": pairs,
            "unknown": unknown,
            "focus": focus,
            "caps": {
                "words": 64,
                "definitions": 3,
                "relations": 10,
                "inflections": 10,
                "next": 5,
            },
            "disclosures": ["Scan uses compiled segments only.", "Caps are presentation limits only."],
        }

    def _workbook_data(self):
        if self._workbook is not None:
            return self._workbook
        segment = self.lm.get("workbook_segment")
        raw = json.loads(bytes(self.model._segment_bytes(segment)).decode("utf-8"))
        self._workbook = raw
        return raw

    @staticmethod
    def _workbook_for_word(word: str, raw: dict) -> dict:
        groups: dict[str, list[int]] = {}
        for index, attribute in enumerate(raw["attribute_ids"]):
            groups.setdefault(attribute.split(".", 1)[0], []).append(index)
        records = {int(record["ascii"]): record["cells"] for record in raw["records"]}
        group_values = {group: [] for group in groups}
        unresolved = 0
        chars = [char for char in word if 32 <= ord(char) <= 126]
        for char in chars:
            cells = records.get(ord(char), [])
            for group, indexes in groups.items():
                values = [encode_cell(cells[index]) for index in indexes]
                group_values[group].append(float(np.mean([value > 0 for value in values])) if values else 0.0)
            unresolved += sum(
                1 for index in range(len(raw["attribute_ids"]))
                if str(cells[index]).strip() in UNRESOLVED
            )
        return {
            "groups": {group: float(np.mean(values)) if values else 0.0 for group, values in group_values.items()},
            "unresolved_cells": int(unresolved),
        }

    def run(self, prompt: str, max_tokens: int = 40, temperature: float = 0.0, hooks=None) -> dict:
        self.load()
        procedure = "continue"
        text = ""
        tokens: list[int] = []
        provenance = {"type": "compiled_sqlir", "weight_source": self._weights.describe() if self._weights else None}
        definition_word = self._procedure_word(prompt)
        related_word = self._related_word(prompt)
        if definition_word is not None:
            procedure = "define"
            try:
                normalized = norm_word(definition_word)
            except ValueError:
                normalized = ""
            token_id = self._id_for_word(normalized) if normalized else 0
            definitions = [] if token_id == 0 else self._definition_values(token_id)
            if definitions:
                text = "\n".join(f"{index}. {value}" for index, value in enumerate(definitions, 1))
            elif self.lm.get("definition_source") is None:
                text = "No definition source column in this database"
            else:
                text = f"No definition for {definition_word} in {self._source_label()}"
            provenance["source"] = self._source_label()
            tokens = self.tokenize(text)
        elif related_word is not None:
            procedure = "related"
            try:
                normalized = norm_word(related_word)
            except ValueError:
                normalized = ""
            token_id = self._id_for_word(normalized) if normalized else 0
            relations = [] if token_id == 0 else self._relation_values(token_id)
            grouped: dict[str, list[str]] = {}
            for relation_type, target in relations:
                key = "null" if relation_type is None else str(relation_type)
                grouped.setdefault(key, []).append(str(target))
            text = "\n".join(
                f"{key}: {', '.join(values)}" for key, values in sorted(grouped.items())
            )
            if not text:
                text = f"No relations for {related_word}"
            provenance["source"] = "relation"
            tokens = self.tokenize(text)
        else:
            generated, prompt_tokens = self._continuation(prompt, max_tokens, temperature, hooks)
            tokens = generated
            text = self.detokenize(generated)
            provenance["source"] = "bigram/collocation/unigram"
            result_check = self._lexicon_check(generated, prompt_tokens)
        if definition_word is not None or related_word is not None:
            result_check = self._lexicon_check(tokens)
        result = {
            "procedure": procedure,
            "text": text,
            "tokens": tokens,
            "provenance": provenance,
            "lexicon_check": result_check,
            "disclosures": [
                DISCLOSURE,
                f"readout coefficients: {READOUT}",
                "Scan caps are presentation limits only.",
            ],
        }
        result["scan"] = self.scan(prompt)
        return result

    def generate(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0, hooks=None
    ) -> str:
        return self.run(prompt, max_tokens=max_tokens, temperature=temperature, hooks=hooks)["text"]
