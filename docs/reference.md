# Reference

This page documents the public `scpc-python` API.

## Overview

| Function | Description |
|---|---|
| `scpc()` | Run spatial correlation-robust inference for a fitted model |
| `SCPCResult` | Structured result object returned by `scpc()` |

## Conventions

### Import Pattern

The package exports both the function name `scpc` and a callable module object.
Both of the following are supported:

- `from scpc import scpc`
- `import scpc` followed by `scpc(...)`

### Supported Models

`scpc()` accepts fitted:

- `statsmodels` regression results
- `pyfixest` models, including IV specifications

It does not accept `FixestMulti`.

For `pyfixest`, the model must retain its estimation sample. Fit with
`store_data=True` and `lean=False`.

### Data Alignment and Coordinates

- `data` must be the data used to fit the model, or a superset that still
  preserves the fitted row identities
- for `statsmodels`, `data.index` must be unique because model rows are mapped
  back through the index
- provide exactly one coordinate specification:
  - `lon` and `lat` for geographic coordinates
  - `coords_euclidean` for planar coordinates

Do not pass both coordinate specifications in the same call.

### Clustering and Method Selection

- `cluster` names an optional clustering column in `data`
- when coordinates vary within clusters, `scpc-python` uses the first
  observation's coordinates for each cluster and emits a warning
- when a model uses absorbed fixed effects and external clustering, the
  conditional adjustment is not implemented; use `uncond=True`
- `method="auto"` selects the exact branch for smaller problems and the
  approximation branch once the large-`n` threshold is reached

## Core Inference

### `scpc()`

Run spatial correlation-robust inference.

**Signature**

```python
scpc(
    model,
    data,
    *,
    lon=None,
    lat=None,
    coords_euclidean=None,
    cluster=None,
    ncoef=None,
    avc=0.03,
    method="auto",
    large_n_seed=1,
    uncond=False,
    cvs=False,
)
```

**Parameters**

- `model`: fitted model object. Supported classes include `statsmodels`
  regression results and `pyfixest` fits, including IV specifications.
- `data`: data frame used to fit `model`, with the coordinate columns and any
  clustering variable.
- `lon`, `lat` (`str | None`): longitude and latitude column names for
  geographic distance calculations.
- `coords_euclidean` (`Sequence[str] | None`): Euclidean coordinate column
  names. Use instead of `lon` and `lat`.
- `cluster` (`str | None`): optional clustering column in `data`.
- `ncoef` (`int | None`): number of coefficients to report. `None` reports all
  coefficients.
- `avc` (`float`): upper bound on average pairwise correlation. Must lie in
  `(0.001, 0.99)`. Default: `0.03`.
- `method` (`str`): spatial algorithm.
  - `"auto"`: choose between exact and approximate branches
  - `"exact"`: always use the full distance matrix
  - `"approx"`: always use the large-`n` approximation branch
- `large_n_seed` (`int`): seed used by the large-`n` approximation branch.
  Ignored when the exact branch is used.
- `uncond` (`bool`): if `True`, skip the conditional adjustment and report
  unconditional critical values only.
- `cvs` (`bool`): if `True`, store per-coefficient critical values at the 32%,
  10%, 5%, and 1% levels.

**Returns**

- `SCPCResult`

## Return Object

### `SCPCResult`

Returned by `scpc()`.

**Fields**

- `scpcstats`: main output table with estimates, standard errors,
  t-statistics, p-values, and 95% interval endpoints
- `scpccvs`: stored critical values, or `None` when `cvs=False`
- `w`: final spatial projection matrix used for inference
- `avc`
- `c0`
- `cv`
- `q`
- `method`
- `large_n_seed`
- `call`

**Notes**

- `scpcstats` has one row per reported coefficient and six columns:
  coefficient estimate, standard error, t-statistic, p-value, lower bound, and
  upper bound
- `method` records the spatial algorithm actually used: `"exact"` or
  `"approx"`
- the stable access path in the current package is through these stored arrays
  and metadata fields

The type also declares `__str__()`, `summary()`, `coef()`, and `confint()`
methods. In the current package state, the result object should be treated as a
field-oriented container rather than relying on those helpers.
