# Zero-walk integer search

`zerowalk` is a CPU-only, NumPy-only search method for integer weights over the
canonical 95 × 437 ASCII workbook. It reads labels only from supplied
ground-truth text/records, and a comparison such as “did this coordinate move
improve the loss?” is the walk's only search decision. It does not synthesize
training data or assign values to unresolved workbook cells.

## Windows commands

Run these from the repository root with Python 3.12:

```powershell
py -3.12 -m pip install numpy
# Optional; required only when using --baselines
py -3.12 -m pip install scikit-learn
py -3.12 -m zerowalk audit --workbook sqlir\conical437_real_workbook.json
py -3.12 -m zerowalk fit --workbook sqlir\conical437_real_workbook.json --text webster.txt --offsets 1 --out run_k1 --baselines --certify
py -3.12 -m zerowalk fit --workbook sqlir\conical437_real_workbook.json --text webster.txt --offsets 2 --out run_k2 --baselines --certify
```

Use `--records records.jsonl` instead of `--text webster.txt` for
`compile_weights.py`-style JSONL records. `--baselines` lazily imports
scikit-learn for the optional multinomial logistic-regression baseline; it is
not a package requirement. The workbook and corpus are hashed into
`report.json`; only their basenames and SHA-256 digests are recorded, not local
absolute paths. `model.npz` contains integer `W`, feature names, the profile
table and ids, offsets, and (when requested) the two certificate arrays.
`--stop-acc FLOAT` stops when count-weighted train accuracy reaches the chosen
threshold (default: unset); `--patience N` defaults to 3 sweeps without a
strict train-accuracy improvement. This is a functional stop rule, not a
requirement to reach zero loss. Zero loss, a coordinate-wise optimum, and
`--max-sweeps` remain additional stop conditions.

## Encoding and samples

Columns whose interaction is `KEY` are excluded; column 0 remains a label and
is never a feature. For each remaining cell, string `"1"`/`"0"` becomes
`+1`/`−1` in an emitted boolean feature. Status cells `E/K/M/R/S/U/V/X`
contribute zero. Every other supplied value is one-hot encoded per column and
string value; unresolved cells stay unresolved. The canonical result is
`F.shape == (95, 94)`, with 21 boolean and 73 one-hot features, 26 distinct
row profiles, and rank 26.

At every ground-truth position, the target is that known character and the
context is the preceding `K` characters. Missing prefix positions use an
absent profile distinct from all 26 workbook profiles. The context tuple is
therefore a sufficient statistic. Samples are aggregated by integer-encoded
context and target using `numpy.unique`; the first 80% of accepted records
train the model and the last 20% are held out without shuffling. Each feature
row is a bias followed by the 94 profile features at each offset.

The eight status letters carry no feature contribution. In particular, the
profile groups printed by `audit` are **indistinguishable from defined
evidence**. Separating them requires binding `E/R/M/V` cells. Do not invent
those values.

## Search and evaluation

The fit is a Gauss–Seidel coordinate search on an integer 95 × D matrix. Its
objective is the count-weighted, unit-margin Crammer–Singer hinge loss. For
each class-coordinate line, the readout evaluates only rows where that input
coordinate is nonzero. `line_argmin` walks whichever direction initially
improves, gallops over improving integer steps, then bisects the monotone
improvement edge. `walk_edge` probes `1, 3, 7, …, 2^(k+1)−1`; for an edge at
integer `T`, the exhaustive test fixture uses
`2 * floor(log2(T + 1)) + 1` total probes (gallop plus bisection). All loops
are exact `int64` computations and progress/ETA is sent to stderr.

The user-provided `zero_walk.py` is copied byte-for-byte as
`zerowalk/zero_walk.py`; both original `bracket_walk` definitions remain
unchanged, so the second definition wins exactly as in the source. The
vectorized `walk_known` follows the original `zero_walk.walk` gallop/cross/
halve trace and accumulates the displacement by summing signed steps. This
package extends that comparison-only walk to discrete weight search; it is
not a gradient optimizer.

Reported comparisons include the walk readout, optional sklearn baseline,
feature-limited per-context majority bound, and global majority class. The
feature-limited train score is the maximum any function of the encoded
contexts can attain on those aggregated training samples. Optional strict
margin certification reports prediction-invariant integer slack in each
coordinate direction. A reported weight interval containing zero means that
weight can be set to zero **individually, not jointly**.

## Boundaries

- `walk_edge` and the line search require a monotone one-crossing predicate;
  `line_argmin`'s exactness assumes a convex integer line objective.
- A coordinate-wise optimum is not a global optimum when the loss remains
  above zero.
- The model and comparisons use exact `int64`; integer overflow remains a
  precision floor for extreme weights or accumulated scores.
- Workbook profiles that match under defined evidence remain grouped. Missing
  `E/R/M/V` bindings cannot be inferred from this package.
- Test-only piecewise-linear fixtures are software fixtures, not observations
  or training data. Benchmark reports use the supplied real corpus and include
  SHA-256 provenance.
