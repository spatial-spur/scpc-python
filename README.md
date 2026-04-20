![Tests](https://github.com/DGoettlich/scpc-python/actions/workflows/test.yaml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)

```text
███████  ██████ ██████   ██████       ██████  ██    ██ ████████ ██   ██  ██████  ███    ██ 
██      ██      ██   ██ ██            ██   ██  ██  ██     ██    ██   ██ ██    ██ ████   ██ 
███████ ██      ██████  ██      █████ ██████    ████      ██    ███████ ██    ██ ██ ██  ██ 
     ██ ██      ██      ██            ██         ██       ██    ██   ██ ██    ██ ██  ██ ██ 
███████  ██████ ██       ██████       ██         ██       ██    ██   ██  ██████  ██   ████ 
                                                                                           
```

# scpc-python: Spatial Correlation-Robust Inference

`scpc-python` provides spatial correlation-robust inference for regression
coefficients following Müller and Watson (2022, 2023). 

## Installation

You can install `scpc-python` from PyPI (coming soon) with: 

```python
uv venv --python 3.11
uv pip install scpc-python
```

or from GitHub with:

```python
uv venv --python 3.11
uv pip install git+https://github.com/DGoettlich/scpc-python.git
```

## Basic Usage

This example starts from the transformed branch of the Becker, Boll, and Voth (2026)
workflow. In the full workflow, you would first use `spurtest` to decide
whether to stay in levels or transform. Here we assume the transformed branch
and move directly to the regression plus `scpc` step.

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
```

`scpc()` returns an `SCPCResult` object:

- `result.scpcstats`: the main inference table with coefficient estimates,
  standard errors, t statistics, p values, and 95% interval endpoints
- `result.scpccvs`: optional stored critical values at 32%, 10%, 5%, and 1%
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
- `avc`: upper bound on the average pairwise correlation
- `uncond`: whether to skip the conditional adjustment
- `cvs`: whether to store the extra critical values
