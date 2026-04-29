![Tests](https://github.com/DGoettlich/scpc-python/actions/workflows/test.yaml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)

<p align="center">
  <img src="assets/logo.png" alt="SPUR logo">
</p>

# scpc-python

``scpc-python` provides spatial correlation-robust inference for regression coefficients following Müller and Watson (2022, 2023), implemented in Python based on their [original Stata implementation](https://github.com/ukmueller/SCPC).

**When using this code, please cite [Becker, Boll and Voth (2026)](https://pauldavidboll.com/SPUR_Stata_Journal_website.pdf):**

```bibtex
@Article{becker2026,
  author    = {Becker, Sascha O. and Boll, P. David and Voth, Hans-Joachim},
  title     = {Testing and Correcting for Spatial Unit Roots in Regression Analysis},
  journal   = {Stata Journal},
  year      = {forthcoming},
  note      = {Forthcoming}
}
```

If you encounter any issues or have any questions, please open an issue on GitHub or contact the authors.

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
- `avc`: upper bound on the average pairwise correlation
- `uncond`: whether to skip the conditional adjustment
- `cvs`: whether to store the extra critical values

## Documentation

Please refer to [the package documentation](https://spatial-spur.github.io/scpcR/) for detailed information and other (R, Python, Stata) packages.

## References

Becker, Sascha O., P. David Boll and Hans-Joachim Voth "Testing and Correcting for Spatial Unit Roots in Regression Analysis", Forthcoming at the Stata Journal.

Müller, Ulrich K. and Mark W. Watson "Spatial Correlation Robust Inference", Econometrica 90(6) (2022), 2901–2935. https://www.princeton.edu/~umueller/SHAR.pdf.

Müller, Ulrich K. and Mark W. Watson "Spatial Correlation Robust Inference in Linear Regression and Panel Models", Journal of Business & Economic Statistics 41(4) (2023), 1050–1064. https://www.princeton.edu/~umueller/SpatialRegression.pdf.
