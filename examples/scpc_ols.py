from __future__ import annotations
import pandas as pd
import statsmodels.formula.api as smf
import scpc


if __name__ == "__main__":
    data = pd.DataFrame(
        {
            "y": [1.0, 1.8, 2.9, 3.7, 5.1],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "lat": [0.0, 1.0, 0.5, 1.5, 2.0],
            "lon": [0.0, 0.0, 1.0, 1.0, 1.5],
        }
    )
    fit = smf.ols("y ~ x", data=data).fit()

    result = scpc.scpc(
        fit,
        data=data,
        lat="lat",
        lon="lon",
    )

    print(result)
