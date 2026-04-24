# scpc-python

`scpc-python` provides the SCPC inference stage of the SPUR workflow for
cross-sectional regressions with spatial dependence. For the diagnostic and
transformation stage in Python, see `spur-python`.

## Installation

The package can be installed as standalone package:

```bash
uv pip install scpc-python
```

or as `scpc-python` dependency with:

```bash
uv pip install spur-python
```

## Example: Chetty Dataset

In this example, we walk you through the SCPC inference workflow step-by-step.
SCPC is the inference stage of the SPUR workflow, so we start from a fitted
regression model. We also provide a one-stop [pipeline wrapper](#pipeline-wrapper)
if you want to run the full SPUR workflow in one step.

### Data preparation

For illustration, we load the Chetty dataset from `spur-python`. Of course, the
analysis in principle follows the same logic on any other dataset. In this
specific case, we first omit the non-contiguous US states. We also drop rows
with missing values.

```python
from spur import load_chetty_data

df = load_chetty_data()

df = df[~df["state"].isin(["AK", "HI"])][
    ["am", "gini", "fracblack", "lat", "lon"]
].copy()

df = df.dropna(subset=["am", "gini", "fracblack", "lat", "lon"])
```

### Fitting the regression

SCPC needs a fitted model, the data used to fit that model, and spatial
coordinates supplied either as `lon` / `lat` or as Euclidean coordinates.

```python
import statsmodels.formula.api as smf

fit = smf.ols("am ~ gini + fracblack", data=df).fit()
```

### Running SCPC inference

We suggest applying SCPC inference after estimating the regression:

```python
from scpc import scpc

out = scpc(
    fit,
    data=df,
    lon="lon",
    lat="lat",
)

print(out)
```

### Interpreting the output

`print(out)` shows an R-like SCPC inference table. From `scpc-python>=0.1.2`,
you can also use `out.coef()`, `out.confint()`, and `out.summary()` for named
access. The raw arrays remain available as `out.scpcstats` and, when
`cvs=True`, `out.scpccvs`.

If your coordinates are Euclidean rather than geographic, use
`coords_euclidean=[...]` instead of `lon` and `lat`.

### Pipeline wrapper

As a shortcut to implementing the full SPUR workflow manually, use the
`spur-python` pipeline wrapper. It runs the diagnostics, transformation, and
SCPC inference steps in one call.

```python
import spur

result = spur(
    "am ~ gini + fracblack",
    data=df,
    lon="lon",
    lat="lat",
)
```

The nested SCPC results can be printed directly:

```python
print(result.fits.levels.scpc)
print(result.fits.transformed.scpc)
```

## Next Step

See [Reference](reference.md) for the full public API, parameter meanings, and
return objects.
