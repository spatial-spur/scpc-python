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

`scpc-python` needs:

- a fitted model
- the data used to fit that model
- spatial coordinates supplied either as `lon` / `lat` or as Euclidean
  coordinates

The example below uses `load_chetty_data()` from `spur-python`.

```python
from scpc import scpc
from spur import load_chetty_data
import statsmodels.formula.api as smf

df = load_chetty_data()

df = df[~df["state"].isin(["AK", "HI"])][
    ["am", "gini", "fracblack", "lat", "lon"]
].copy()

df = df.dropna(subset=["am", "gini", "fracblack", "lat", "lon"])

fit = smf.ols("am ~ gini + fracblack", data=df).fit()

out = scpc(
    fit,
    data=df,
    lon="lon",
    lat="lat",
)

print(out)
```

- `fit` is the fitted model
- `data` is the underlying data frame
- `lon` and `lat` identify the coordinate columns

If your coordinates are Euclidean rather than geographic, use
`coords_euclidean=[...]` instead of `lon` and `lat`.

`print(out)` shows an R-like SCPC inference table. From
`scpc-python>=0.1.2`, use `out.coef()`, `out.confint()`, and `out.summary()`
for named access. The raw arrays remain available as `out.scpcstats` and, when
`cvs=True`, `out.scpccvs`.

If you need the diagnostic and transformation stage before inference, the
easiest entry point is `spur-python`, which uses `scpc-python` internally for
the inference step.

## spur-python integration

`scpc-python` is also a dependency of the `spur-python` package, so in
particular, you can apply `scpc` as part of the pipeline using:

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
