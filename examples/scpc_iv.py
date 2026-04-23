from __future__ import annotations
import numpy as np
import pandas as pd
import pyfixest as pf
import scpc


if __name__ == "__main__":
    rng = np.random.default_rng(2001)
    n = 120
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.9 * z + 0.4 * w + 0.7 * u + rng.normal(scale=0.2, size=n)
    y = 1.0 + 1.2 * x + 0.5 * w + u
    data = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "w": w,
            "z": z,
            "coord_x": rng.uniform(size=n),
            "coord_y": rng.uniform(size=n),
        }
    )

    fit = pf.feols("y ~ w | x ~ z", data=data)

    result = scpc.scpc(
        fit,
        data=data,
        coords_euclidean=("coord_x", "coord_y"),
        avc=0.1,
        method="exact",
        cvs=True,
    )

    print(result)
