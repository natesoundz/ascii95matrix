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
py -3.12 -m zerowalk discover --workbook sqlir\conical437_real_workbook.json --text webster.txt --offsets 1 --out discover_k1
py -3.12 -m zerowalk discover --workbook sqlir\conical437_real_workbook.json --text webster.txt --offsets 2 --out discover_k2
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
default `--loss softmax` objective is
`Σ nᵢ (logsumexp(u·sᵢ) − u·sᵢ,y)`, with exact `int64` scores
`s = X Wᵀ`, float64 evaluation, and `u = 2**-unit_bits` (default
`--unit-bits 4`). For each class-coordinate line, the readout caches the
log-sum-exp over the other classes for that class sweep and evaluates only
rows where that input coordinate is nonzero. `--loss hinge` selects the
count-weighted, unit-margin Crammer–Singer hinge objective used by the first
version of this package. `line_argmin` memoizes objective probes, walks
whichever direction initially improves, gallops over improving integer steps,
then bisects the monotone improvement edge. Float64 rounding can make the
softmax line objective flat or noisy within about `1e-12` relative near its
minimum. `walk_edge` probes `1, 3, 7, …, 2^(k+1)−1`; for an edge at integer
`T`, the exhaustive test fixture uses `2 * floor(log2(T + 1)) + 1` total
probes (gallop plus bisection). Progress/ETA is sent to stderr.

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

### Joint walk per sweep

`--strategy coordinate` remains the default Gauss–Seidel fit. Use
`--strategy joint --workers N` to propose a Jacobi-style update each sweep;
`--workers 1` avoids a process pool, while larger values create a fresh
class-proposal pool per sweep.

1. Freeze the exact score matrix and find each class-feature proposal with the same line objective as coordinate fit.
2. Assemble proposals in class order; parallelize classes when `--workers` is greater than one.
3. Search along the combined proposal, halving it until the full loss decreases; use float64 score matmul only when the absolute bound is below `2**53`.
4. If no halved proposal improves loss, run one ordinary Gauss–Seidel sweep; retain the same stopping rules and per-sweep loss/accuracy metrics.

```sh
python -m zerowalk fit --workbook sqlir/conical437_real_workbook.json \
  --text webster.txt --offsets 2 --out out_joint \
  --strategy joint --workers 8
```

Webster fit comparison (top-1 percentages; coordinate values are from the
committed reports; `nproc` was 8):

| K | Strategy | Train / test top-1 | Sweeps | Fit seconds | Fallbacks |
|---:|---|---:|---:|---:|---:|
| 1 | coordinate | 17.2587 / 17.4269 | 7 | 4.125 | 0 |
| 1 | joint, workers=1 | 15.2753 / 15.1554 | 7 | 14.309 | 0 |
| 1 | joint, workers=8 | 15.2753 / 15.1554 | 7 | 2.777 | 0 |
| 2 | coordinate | 19.3618 / 19.6708 | 7 | 20.263 | 0 |
| 2 | joint, workers=1 | 17.9718 / 18.4393 | 8 | 67.593 | 0 |
| 2 | joint, workers=8 | 17.9718 / 18.4393 | 8 | 10.661 | 0 |
| 3 | coordinate | 20.5159 / 20.7719 | 15 | 372.829 | 0 |
| 3 | joint, workers=1 | not run after K=3 cutoff | — | — | — |
| 3 | joint, workers=8 | not completed (45-minute cutoff; proposal phase reached sweep 13) | — | — | — |

The K=1/K=2 joint test accuracy is about 1.2–2.3 percentage points below
coordinate fit, despite faster workers=8 fits; the one-worker joint fits were
slower than coordinate. K=3 did not produce a complete joint report within
the time limit, so only the completed workers=8 K=1/K=2 reports are committed.

## Discovering occurrence-bound distinctions

`discover` adds one occurrence indicator per offset for every character in a
multi-member workbook profile, while keeping the ordinary workbook profile
features. Its context keys retain the actual preceding characters (plus an
absent value), so only real corpus occurrences can distinguish members. The
canonical multi-member groups contain 78 member characters across 9 groups
(69 total merges); the set size is derived from the groups rather than
hard-coded.

Each group's merge tree starts with singleton characters. A candidate merge
copies one cluster representative's occurrence-weight blocks to the other
cluster without refitting; the lower drop in count-weighted train correct
positions wins. The tree records every merge cost and accuracy. The
`--tolerance τ` partition applies merges in recorded order while
`cost_positions <= τ * train_positions`, stopping at the first larger cost.
The output reports that partition's train/test accuracy and clusters.

For the uppercase and lowercase alphabet groups, `discover.json` also compares
the final tree split with the five most-agreeing columns from
`legacy/ASCII95_254_BINARY_MATRIX.tsv`, allowing either polarity. This is an
**external reference, not used in fit**; it does not add features, labels, or
constraints to the discovered model.

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

## Single-file walk (`zero_walk_combined.py`)

`zero_walk_combined.py` is a standalone version containing the original walk,
vectorized walks and marbles, integer softmax/hinge readout, and strict-margin
certification. It uses only the Python standard library and NumPy; it does not
import this package. `zero_walk.py` is Nate's original, kept byte-identical as
the parity reference.

From the repository root on Windows:

```powershell
py -3.12 -m pip install numpy
py -3.12 zerowalk\zero_walk_combined.py selftest
py -3.12 zerowalk\zero_walk_combined.py run --old old.json --new new.json --out delta.json
py -3.12 zerowalk\zero_walk_combined.py fit --rows rows.json --out model.json --certify
```

The `fit` command accepts a JSON object with `X` as a rectangular matrix,
`y` as one integer class label per row, optional positive integer `counts`
(default 1 per row), and optional integer `classes` (default `max(y) + 1`).
`X` must contain integers unless `--unit U` is supplied; then values are
quantized with `round(x / U)`.

The committed Webster reports show softmax walk test top-1 accuracies of
17.43%, 19.67%, and 20.77% for K=1/2/3, versus logistic-regression 17.43%,
19.87%, and 20.78%; the hinge K1 walk reached 14.88%. Discovery fit test
accuracy was 24.4% at K1 and 32.2% at K2. The tolerance-zero partition reports
24.4087% and 32.2768% respectively (the latter rounds to 32.3%). The K1
uppercase top split is `BCDFGHJKLMNPQRSTVWXYZ | AEIOU`; its legacy consonant
and vowel-related columns agree at 1.0. This is an external reference, not
used in fit.
