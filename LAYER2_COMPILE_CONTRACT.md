# Layer Two Compile Contract

Layer two is not an independently invented second transformer block. It is
compiled only after layer one exists as a real frozen target-free artifact.

## Entry condition

Layer one must already have passed:

- strict 437-dimensional evidence binding,
- embedding verification and freeze,
- target-free attention verification and freeze,
- FFN residual verification and freeze,
- exact serialization checks,
- target-free replay without access to the answer.

If those conditions are not met, layer two remains blocked.

## Layer-two input

For runtime occurrence (i), layer one returns:

[
h^{(1)}_F(i)
]

plus its candidate/readout field and any explicit unresolved relational
requirements.

Layer two begins from that state. It does not restart from the raw target
embedding.

[
h^{(2)}_0 = h^{(1)}_F
]

If an additional context-conditioned state constructor is declared, it may use
only information available at inference time.

## Compilation evidence

During compilation the known truth may be used to measure what layer one failed
to satisfy.

Define the layer-one remaining requirement:

[
R^{(2)}_i =
T_i - M(h^{(1)}_F(i))
]

where (T_i) is declared ground truth and (M) is the corresponding declared
measurement/constraint operation.

The purpose of layer two is to compile transformations that reduce this
remaining requirement field without changing frozen layer-one parameters.

## Order inside layer two

```text
FROZEN LAYER-ONE OUTPUT
        |
        v
L2 CONTEXT FIELD
        |
        v
L2 ATTENTION RESIDUAL
        |
        v
hA2 = h0_2 + delta_A2
        |
        v
FREEZE L2 ATTENTION
        |
        v
REPLAY REMAINING REQUIREMENTS
        |
        v
COMPILE L2 FFN RESIDUAL
        |
        v
hF2 = hA2 + delta_F2
        |
        v
FREEZE L2 FFN
        |
        v
READOUT / NEXT STATE
```

As with layer one, attention is compiled first and frozen. The FFN is compiled
only from the residual left by the frozen attention stage.

## Prohibitions

Layer two must not:

- read target (y) at runtime,
- modify layer-one frozen parameters,
- use a new arbitrary embedding geometry to bypass layer-one state,
- replace missing evidence with zero,
- hide contradictions by averaging them into a successful result,
- claim improvement without replaying the same declared requirements through
  the frozen layer-one + layer-two path.

## Verification

The layer-two report must include:

- exact layer-one artifact hashes consumed,
- layer-two evidence/constraint count,
- input residual RMS by requirement family,
- post-L2-attention residual,
- post-L2-FFN residual,
- per-family solved / unresolved / conflicting counts,
- exact tensor serialization checks,
- parameter allocation added by layer two,
- complete cumulative parameter allocation,
- target-free replay results,
- `runtime_target_input: false`.

## Architectural purpose

Layer depth is therefore derived rather than arbitrary:

[
	ext{next layer} =
	ext{transformation required by residual constraints left by prior frozen layer}
]

If layer one already resolves a requirement, layer two receives no reason to
rewrite it. If layer two cannot improve a requirement using inference-available
information, that unresolved requirement is reported rather than hidden.
