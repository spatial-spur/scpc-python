![Tests](https://github.com/DGoettlich/scpc-python/actions/workflows/test.yaml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)

<p align="center">
  <img src="assets/logo.png" alt="SPUR logo">
</p>

# scpc-python

`scpc-python` provides spatial correlation-robust inference for regression coefficients following Müller & Watson 2022 and Müller & Watson 2023, implemented in Python based on their [original Stata implementation](https://github.com/ukmueller/SCPC).

**Citation:** If you use this package, please cite Becker, Boll and Voth 2026, Müller & Watson 2022, and Müller & Watson 2023. See [CITATION.bib](CITATION.bib) for the BibTeX entries.

If you encounter any issues or have any questions, please open an issue on GitHub or contact the authors.

## Installation

The package can be installed as standalone package:

```bash
uv pip install scpc-python
```

or as `scpc-python` dependency with:

```bash
uv pip install spur-python
```

Conditional OLS calibration uses the fitted design and the projection order in
Müller and Watson (2023), §5.1. SCPC supports unweighted OLS and PyFixest
2SLS fits. Weighted and other model types raise an error on every backend.

Use `backend="auto"` (the default), `"rust"`, or `"fallback"` to select
execution. `SPATIAL_BACKEND` applies when you omit the argument. The result's
`backend` reports the backend that ran, and `fallback_reason` records an
automatic change. Set `nthreads=1` for a single Rust worker. Linear algebra
providers manage their internal threading. In multiprocessing code that
initializes native threads before starting workers, use the `spawn` or
`forkserver` start method.

With `backend="rust"`, Python aligns the estimation sample, coordinates and
fitted model inputs. Rust aggregates cluster scores, absorbs one-way fixed
effects and builds the conditional OLS or IV projection for exact and Nyström
SCPC. Conditional models with multiple absorbed fixed effects use the Python
implementation under `backend="auto"`.
Each observation in the estimation sample needs a cluster label when clustering
is requested. Coordinates must be identical within each cluster.

The Rust backend requires SCPC API 7 and methods spec 0.3.9 or newer. Install
the adapter from the matching shared-core checkout.

```bash
uv pip install ../spatial-rust/crates/spatial-py
```

Approximate SCPC raises `scpc.InvalidSampledCovarianceError` if its sampled
covariance matrix is asymmetric or materially indefinite. For samples that fit
in memory, request `method="exact"` to compute the unsampled covariance.
Both methods approximate the worst-case calibration over the decay parameter
using a finite grid.
For a given seed, all conditional coefficients use the same sampled site pairs.
Reordering regressors or limiting the number of reported coefficients preserves
the inference for each retained coefficient.

## Basic Usage

This example fits a prespecified transformed regression from the Becker, Boll
and Voth 2026 workflow. `spurtest` can inform a comparison of levels and
transformed results. The coverage arguments for each prespecified analysis
require separate justification after selecting a branch using the same data.

```python
# assumes spur-python is installed
from spur import load_chetty_data, spurtransform
from scpc import scpc
import statsmodels.formula.api as smf

data = load_chetty_data()

data = data[~data["state"].isin(["AK", "HI"])][
    ["am", "gini", "fracblack", "lat", "lon"]
]
data = data.dropna(subset=["am", "gini", "fracblack", "lat", "lon"]).copy()

transformed = spurtransform(
    "am ~ gini + fracblack",
    data,
    lon="lon",
    lat="lat",
    transformation="lbmgls",
    prefix="h_",
)

fit = smf.ols("h_am ~ h_gini + h_fracblack", data=transformed).fit()

result = scpc(
    fit,
    data=transformed,
    lon="lon",
    lat="lat",
    cvs=True,
)

print(result)
```

`scpc()` returns an `SCPCResult` object:

- `print(result)`: prints an R-like SCPC inference table
- `result.scpcstats`: the main inference table with coefficient estimates,
  standard errors, t statistics, p values, and 95% interval endpoints
- `result.scpccvs`: optional stored critical values at 32%, 10%, 5%, and 1%
- `result.coef()`: returns named coefficient estimates in `scpc-python>=0.1.2`
- `result.confint()`: returns named confidence intervals in `scpc-python>=0.1.2`
- `result.summary()`: prints the main table plus confidence intervals in
  `scpc-python>=0.1.2`
- `result.avc`: the average pairwise correlation bound used in the analysis
- `result.c0`: the kernel scale implied by `avc`
- `result.cv`: the unconditional 5% critical value
- `result.q`: the number of retained non-constant spatial principal components

## Key Arguments

The most important `scpc()` arguments in the workflow above are:

- `model`: the fitted regression model
- `data`: the data frame
- `lon`, `lat`: the geodesic coordinate column names
- `coords_euclidean`: use this instead of `lon` / `lat` when coordinates are
  Euclidean rather than geographic
- `cluster`: optional clustering column
- `ncoef`: how many coefficients to report
- `avc`: upper bound on the average pairwise correlation of the calibration benchmark
- `uncond`: whether to skip the conditional adjustment
- `cvs`: whether to store the extra critical values

## Documentation

Please refer to [the package documentation](https://spatial-spur.github.io/scpcR/) for detailed information and other (R, Python, Stata) packages.

### Interpreting the intervals

Each interval targets one coefficient. SCPC uses a finite covariance grid to
calibrate the cutoffs, including when `method="exact"` computes the full
distance matrix.

For 2SLS, ordinary SCPC t inference requires strong identification and the
score-process assumptions of Müller and Watson. The default uses the larger of
the unconditional and conditional cutoffs. The conditional Gaussian guarantee
requires the stated error model given the endogenous regressors, instruments,
and controls. See [Müller and Watson (2023), §5.3](https://www.princeton.edu/~mwatson/papers/Mueller_Watson_JBES_2023.pdf#page=15).

For a scalar endogenous regressor `x` and one excluded instrument `z`, an
Anderson–Rubin test of `beta0` regresses `y - beta0 * x` on `z` and the
exogenous controls. Test the coefficient on `z` using SCPC under its stated
conditions for that regression.

```python
from statsmodels.formula.api import ols
from scpc import scpc

def ar_pvalue(beta0, data):
    adjusted = data.assign(y_null=data["y"] - beta0 * data["x"])
    fit = ols("y_null ~ z + w", data=adjusted).fit()
    out = scpc(fit, adjusted, coords_euclidean=["s1", "s2"])
    return out.scpcstats[out.coef_names.index("z"), 3]
```

Here `w` denotes a control and `s1` and `s2` hold the coordinates. Retain
candidate values with `ar_pvalue(beta0, data) > 0.05` for a nominal 95%
confidence set. The set can be disconnected or unbounded.

## References

Becker, Sascha O., P. David Boll and Hans-Joachim Voth "Testing and Correcting for Spatial Unit Roots in Regression Analysis", The Stata Journal, forthcoming. https://pauldavidboll.com/SPUR_Stata_Journal_website.pdf.

Müller, Ulrich K. and Mark W. Watson "Spatial Correlation Robust Inference", Econometrica 90(6) (2022), 2901–2935. https://www.princeton.edu/~umueller/SHAR.pdf.

Müller, Ulrich K. and Mark W. Watson "Spatial Correlation Robust Inference in Linear Regression and Panel Models", Journal of Business & Economic Statistics 41(4) (2023), 1050–1064. https://www.princeton.edu/~umueller/SpatialRegression.pdf.

Conditional SCPC accounts for absorbed fixed effects in the projection. IV
conditional weights use the transpose of the IV residual maker, including for
clustered data. These corrections can change conditional p-values and critical
values from earlier releases. Multiway absorbed effects use the fallback backend;
requesting the Rust backend for these models raises `NotImplementedError`.
