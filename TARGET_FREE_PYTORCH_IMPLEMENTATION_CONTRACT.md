# Target-Free PyTorch Implementation Contract

This file is the implementation contract for the canonical ASCII95 95 x 437 compiled runtime.

It records the architecture and compile order established after separating **embedding-vector compilation** from the broader idea of "compiling weights."

The authoritative vocabulary geometry is the canonical 95 printable ASCII characters measured across the 437 typed relational dimensions. The registry defines what relationships are available to measure and how participation is gated. It does **not** directly prescribe numerical embedding coordinates.

## 1. Terminology

### Compile the embedding vectors

"Compile the embedding vectors" means:

> Deterministically calculate the numerical value of every embedding dimension for every vocabulary item so that the resulting geometry represents the valid ground-truth relational evidence supplied by the dataset.

This is not conventional model training.

It is not permission to silently manufacture Q/K/V, FFN, decoder, or other parameters.

Those later stages are compiled separately and only after the prior stage has been frozen.

### Ground-truth requirement

For any declared ground-truth relation (T_j) measured from the compiled geometry by (M_j(E)), the objective is:

[
M_j(E) = T_j
]

for every simultaneously satisfiable declared constraint.

If the evidence is contradictory or the constraints cannot all be satisfied, the compiler must report the unresolved conflict. It must not silently average the contradiction away and report success.

## 2. Canonical embedding compilation

The canonical registry has:

- vocabulary size: 95 printable ASCII characters
- relational dimensions: 437
- embedding geometry: (E \in \mathbb{R}^{95 \times 437})

Occurrence-level evidence is represented as:

[
g_{i,k}
]

where occurrence (i) has observed correct character (y_i=c).

Character evidence is accumulated as:

[
G_{c,k}
=
\sum_{i:y_i=c}
w_i g_{i,k}
]

The current relational compiler derives:

[
H_k = \sum_c \phi(G_{c,k})
]

[
\lambda_k = \frac{1}{H_k}
]

[
\rho_{c,k}=\lambda_k\phi(G_{c,k})
]

The field center is:

[
\mu_k=
\frac{
\sum_c w_{c,k}G_{c,k}
}{
\sum_c w_{c,k}
}
]

and:

[
\sigma_k=
\sqrt{
\frac{
\sum_c w_{c,k}(G_{c,k}-\mu_k)^2
}{
\sum_c w_{c,k}
}}
]

The frozen token geometry is:

[
E_{c,k}
=
\frac{G_{c,k}-\mu_k}
{\sigma_k+\epsilon}
]

This normalization is compiled statistics, not learned LayerNorm parameters.

The compiled embedding contains:

[
95 \times 437 = 41,515
]

numerical values.

## 3. Required compile order

The stages must be compiled in this order:

1. Bind the typed registry, participation gates, and valid occurrence evidence.
2. Compile the shared vocabulary embedding geometry (E).
3. Verify the geometry and ground-truth constraints that the geometry is responsible for.
4. Freeze (E).
5. Derive the contextual residual requirements left unresolved by the frozen embedding.
6. Compile attention from those remaining requirements.
7. Verify attention against the frozen embedding.
8. Freeze attention.
9. Replay the same ground-truth evidence through the frozen embedding + frozen attention path.
10. Measure the remaining residual.
11. Compile the FFN only against that remaining residual.
12. Freeze the FFN.
13. Run the complete target-free runtime.
14. Produce candidate winner / survivor / eliminated records and residual verification.

No later stage may rewrite an earlier frozen stage simply because its own residual is difficult to solve.

## 4. Residual is the running state

Residual is not a separate model component.

The state progresses as:

[
h_0
\rightarrow
h_A = h_0 + \Delta_A
\rightarrow
h_F = h_A + \Delta_F
]

The embedding establishes the initial geometry.

Attention writes only what the frozen embedding did not resolve.

The FFN writes only what the frozen embedding + frozen attention path did not resolve.

Every stage must report the residual it receives and the residual it leaves.

## 5. Critical target-free runtime rule

Ground-truth target (y) may be used during compilation because it is the known correct answer used to derive requirements.

Ground-truth target (y) must **never** be supplied to runtime inference.

The historical 254-dimensional control is not target-free because it initializes contextual processing from the known target and uses the target when selecting transformations. That path is useful as a numerical control, not as next-character inference.

For the canonical 437-dimensional runtime, the starting state for an unknown slot must come from visible context only.

The runtime API should therefore have no target argument.

A verification report must explicitly contain:

```text
runtime_target_input: false
```

Any runtime code path that reads the answer character before scoring is leakage and must fail verification.

## 6. Context-only candidate field

For visible relation offset (o), context character (c), and candidate (y), compile the mutual relational specificity:

[
r_{o,c,y}
=
\sqrt{
P(y\mid o,c)
P(c\mid o,y)
}
]

For the current fourteen offsets:

```text
-7 -6 -5 -4 -3 -2 -1 +1 +2 +3 +4 +5 +6 +7
```

the relation field has shape:

[
14 \times 95 \times 95
]

or:

[
126,350
]

compiled values.

At runtime, combine only visible relations and only gate-eligible candidates.

A context-only candidate score field may be formed from the visible log relations:

[
\ell_y
=
\frac{1}{N}
\sum_j \log r_{j,y}
]

and normalized across eligible candidates:

[
q_y
=
\operatorname{softmax}(\ell)_y
]

where (q \in \mathbb{R}^{95}).

Ineligible candidates must remain excluded rather than being reinterpreted as negative observations.

## 7. Target-free starting state

The runtime must not initialize the unknown slot with (E[y]).

Instead:

[
h_0=qE
]

where:

- (q) is derived only from visible context and gates
- (E) is the already-frozen compiled embedding table

Thus:

[
q[95] @ E[95,437]
\rightarrow
h_0[437]
]

This is the initial state for the unknown slot.

## 8. Attention

The canonical target-free attention stage operates from the context field.

The compiled attention value field is:

[
V_{att}
\in
\mathbb{R}^{14 \times 95 \times 437}
]

with:

[
14 \times 95 \times 437
=
581,210
]

compiled values.

The relation field supplies the matching/specificity information that conventional dense Q/K matrices would normally parameterize.

For visible relation scores (s_j):

[
\alpha_j=\operatorname{softmax}(s_j)
]

Attention writes:

[
\Delta_A
=
\sum_j \alpha_j V_j
]

and the first residual update is:

[
h_A=h_0+\Delta_A
]

Attention must be compiled against the residual left by the frozen embedding.

It must then be frozen before FFN compilation begins.

## 9. FFN

After attention is frozen, replay the ground-truth evidence through:

[
E
\rightarrow
h_0
\rightarrow
h_A
]

During compilation only, the known target (y) may be used to compute the remaining required correction.

The current target-free field-fired FFN representation uses:

[
C_{FFN}
\in
\mathbb{R}^{95 \times 437}
]

or:

[
41,515
]

compiled values.

At runtime:

[
c_{field}=qC_{FFN}
]

The field therefore selects/blends the correction from context rather than indexing a correction with the unknown target.

Using:

[
d=h_0-h_A
]

the FFN write is:

[
\Delta_F
=
c_{field}\odot d
]

and the second residual update is:

[
h_F=h_A+\Delta_F
]

The historical per-target normal-equation solve can remain as a compilation/audit reference, but runtime may not select `corr[y]` using the unknown answer.

The target-free extension must be verified independently.

## 10. Decoder / output head

The decoder should be tied directly to the frozen vocabulary geometry rather than introducing an unrelated learned output projection.

For each eligible candidate (y):

[
s_y
=
-\lVert E_y-h_F\rVert^2
]

Ineligible candidates are masked.

The winner is:

[
\hat y=\arg\max_y s_y
]

The runtime report must preserve:

- winner
- surviving candidates
- eliminated candidates
- score or correspondence evidence for each relevant candidate

The decoder therefore adds no independent parameter matrix in the current design.

## 11. PyTorch representation

PyTorch is the execution substrate. It does not change the compilation method.

The embedding should be loaded as frozen:

```python
embedding = nn.Embedding.from_pretrained(
    compiled_E,
    freeze=True,
)
```

Compiled relation, attention, and FFN tensors should be stored as frozen buffers or non-trainable parameters.

The runtime must have:

```text
trainable_parameters = 0
```

There must be no:

- optimizer
- gradient-based fitting
- epochs
- random initialization used as an unresolved semantic substitute
- conventional training loop

If PyTorch autograd is not required for another diagnostic purpose, inference should run under `torch.no_grad()`.

## 12. Canonical 95 x 437 numerical allocation

Current intended runtime allocation:

| Component | Shape | Compiled values |
| --- | ---: | ---: |
| Frozen embedding (E) | 95 x 437 | 41,515 |
| Mutual relation field | 14 x 95 x 95 | 126,350 |
| Attention value field | 14 x 95 x 437 | 581,210 |
| FFN correction field | 95 x 437 | 41,515 |
| Learned normalization parameters | none | 0 |
| Independent decoder parameters | none; tied to (E) | 0 |
| **Total compiled runtime values** | | **790,590** |

Allocation percentages:

- embedding: approximately 5.25%
- mutual relation field: approximately 15.98%
- attention values: approximately 73.52%
- FFN correction field: approximately 5.25%
- trainable parameters: 0

In PyTorch bookkeeping, the frozen `nn.Embedding` may still appear in `model.parameters()`, but `requires_grad=False`.

Therefore report both:

```python
total_parameter_objects = sum(p.numel() for p in model.parameters())

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

and separately report the complete frozen numerical allocation, including buffers.

## 13. Required verification

A successful canonical compile/runtime report must verify at least:

### Embedding

- shape is exactly 95 x 437
- all values finite
- every coordinate recomputes from the frozen evidence/statistics
- registry participation gates are obeyed
- no historical 254-dimensional sign forcing is imported
- declared ground-truth geometry constraints are measured after compilation
- unresolved/contradictory constraints are reported

### Attention

- attention compilation occurs only after embedding freeze
- runtime relation scores use visible context only
- no runtime target lookup
- attention writes additive residual only
- post-attention residual is measured
- serialized tensors exactly match verified tensors

### FFN

- FFN compilation occurs only after attention freeze
- compile-time truth may derive the required residual
- runtime correction is fired from the context/candidate field, not (y)
- post-FFN residual is measured
- FFN must not be reported as successful if it worsens or falsely hides unresolved requirements

### Decoder

- candidate eligibility gates are applied before winner selection
- readout uses the frozen geometry
- winner, survivors, and eliminations are recorded
- runtime does not receive the target

### Whole runtime

The runtime path must be:

```text
VISIBLE CONTEXT
    |
    v
RELATIONAL FIELD
    |
    v
GATE-ELIGIBLE CANDIDATES
    |
    v
NORMALIZED q[95]
    |
    v
h0 = q @ E
    |
    v
ATTENTION RESIDUAL
    |
    v
hA = h0 + delta_A
    |
    v
FFN RESIDUAL
    |
    v
hF = hA + delta_F
    |
    v
COMPARE AGAINST ELIGIBLE E ROWS
    |
    v
WINNER / SURVIVORS / ELIMINATED
```

## 14. Required compile report

Each real compile should emit a machine-readable report containing at minimum:

```text
embedding_shape
embedding_values
relation_field_shape
relation_field_values
attention_shape
attention_values
ffn_shape
ffn_values
decoder_additional_values
total_compiled_runtime_values
trainable_parameters
runtime_target_input
embedding_residual_rms
post_attention_residual_rms
post_ffn_residual_rms
embedding_frozen
attention_frozen
ffn_frozen
serialization_exact
ground_truth_constraints_total
ground_truth_constraints_satisfied
ground_truth_constraints_unresolved
ground_truth_constraints_conflicting
```

Do not replace an unresolved field with a guessed success value.

## 15. Status boundary

The repository currently contains:

- the authoritative canonical 95 x 437 registry
- the 437-dimensional relational geometry compiler
- the historical 254-dimensional control compiler and its known-target attention/FFN path

The complete real-data 437-dimensional target-free PyTorch runtime still requires the actual occurrence-level relational evidence bindings and a full compile against those bindings.

A synthetic target-free PyTorch execution test can establish that the execution graph is functional, but it must not be reported as proof that the real 437-dimensional ground-truth field has already been solved.

The next implementation must therefore preserve the distinction between:

1. **architecture/runtime correctness**
2. **real evidence binding correctness**
3. **ground-truth constraint satisfaction**
4. **target-free predictive evaluation**

All four must be reported separately.

---

**Primary rule:** derivation is definition. Compile each stage from explicit evidence and explicit residual requirements, freeze it, verify it, then move to the next stage. Never use the unknown target at runtime.
