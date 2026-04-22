from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCPC_R_GITHUB = "spatial-spur/scpcR"
SCPC_R_REF = "v0.1.3"
SCPC_R_VERSION = "0.1.3"
SCPC_R_TARBALL_URL = (
    f"https://github.com/{SCPC_R_GITHUB}/archive/refs/tags/{SCPC_R_REF}.tar.gz"
)
FIXEST_R_VERSION = "0.14.0"
SANDWICH_R_VERSION = "3.1-1"
SCPC_R_LIBRARY = PROJECT_ROOT / ".pytest_cache" / "scpcR" / SCPC_R_REF / "lib"
R: str | None = shutil.which("Rscript")


def normalize_r_coef_name(name: str) -> str:
    """Normalize R coefficient names to the Python naming convention."""
    if name == "(Intercept)":
        return "Intercept"
    if name.startswith("fit_"):
        return name[4:]
    return name


def reorder_r_rows_to_py(
    py_names: list[str], r_names: list[str], values: np.ndarray
) -> np.ndarray:
    """Reorder an R row-indexed object into Python coefficient order."""
    normalized_r = [normalize_r_coef_name(name) for name in r_names]
    order = [normalized_r.index(name) for name in py_names]
    return np.asarray(values)[order]


def reorder_r_columns_to_py(
    py_names: list[str], r_names: list[str], values: np.ndarray
) -> np.ndarray:
    """Reorder an R column-indexed object into Python coefficient order."""
    normalized_r = [normalize_r_coef_name(name) for name in r_names]
    order = [normalized_r.index(name) for name in py_names]
    values = np.asarray(values)
    if values.ndim == 1:
        # json turns one-column r matrices into flat vectors, so put the
        # column dimension back before reordering.
        values = values[:, None]
    return values[:, order]


def normalize_r_score_names(score_names: list[str], coef_names: list[str]) -> list[str]:
    """Normalize R score column names for parity checks.

    In the pinned `fixest 0.14.0` stack, FE-IV score matrices can carry a
    blank column name even though the score columns still follow coefficient
    order. We fill only those blank slots from the coefficient names and
    reject any real name mismatch.
    """
    if len(score_names) != len(coef_names):
        raise ValueError("R score columns do not line up with the coefficient names.")

    normalized: list[str] = []
    for score_name, coef_name in zip(score_names, coef_names, strict=True):
        if score_name == "":
            # fixest leaves this name blank for fe-iv scores, but the column
            # still matches the coefficient in the same position.
            normalized.append(coef_name)
            continue
        if score_name != coef_name:
            raise ValueError(
                "R score column names do not match the coefficient names."
            )
        normalized.append(score_name)

    return normalized


def reorder_r_square_matrix_to_py(
    py_names: list[str], r_names: list[str], values: np.ndarray
) -> np.ndarray:
    """Reorder an R square matrix on both axes into Python coefficient order."""
    reordered = reorder_r_rows_to_py(py_names, r_names, values)
    return reorder_r_columns_to_py(py_names, r_names, reordered)


def make_basic_iv_data(seed: int, n: int = 120) -> pd.DataFrame:
    """Build a simple IV dataset without fixed effects or clustering."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.9 * z + 0.4 * w + 0.7 * u + rng.normal(scale=0.2, size=n)
    y = 1.0 + 1.2 * x + 0.5 * w + u
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "w": w,
            "z": z,
            "coord_x": rng.uniform(size=n),
            "coord_y": rng.uniform(size=n),
        }
    )


def make_one_way_fe_iv_data(seed: int, n_fe: int = 20, t_per_fe: int = 5) -> pd.DataFrame:
    """Build a one-way absorbed-fe IV dataset."""
    rng = np.random.default_rng(seed)
    n = n_fe * t_per_fe
    fe = np.repeat(np.arange(1, n_fe + 1), t_per_fe)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.7 * z + 0.3 * w + u
    y = 1.0 + 1.1 * x + 0.4 * w + rng.normal(size=n_fe)[fe - 1] + u
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "w": w,
            "z": z,
            "fe": fe,
            "coord_x": rng.uniform(size=n),
            "coord_y": rng.uniform(size=n),
        }
    )


def make_two_way_fe_iv_data(
    seed: int, n1: int = 5, n2: int = 4, reps: int = 4
) -> pd.DataFrame:
    """Build a two-way absorbed-fe IV dataset."""
    rng = np.random.default_rng(seed)
    n = n1 * n2 * reps
    fe1 = np.repeat(np.repeat(np.arange(1, n1 + 1), n2), reps)
    fe2 = np.repeat(np.tile(np.arange(1, n2 + 1), n1), reps)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.7 * z + 0.2 * w + u
    y = 1.0 + 1.1 * x + 0.35 * w + rng.normal(size=n1)[fe1 - 1] + rng.normal(
        size=n2
    )[fe2 - 1] + u
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "w": w,
            "z": z,
            "fe1": fe1,
            "fe2": fe2,
            "coord_x": rng.uniform(size=n),
            "coord_y": rng.uniform(size=n),
        }
    )


def make_clustered_iv_data(
    seed: int, n_cluster: int = 40, cl_size: int = 3
) -> pd.DataFrame:
    """Build a clustered IV dataset with cluster-constant coordinates."""
    rng = np.random.default_rng(seed)
    n = n_cluster * cl_size
    cl = np.repeat(np.arange(1, n_cluster + 1), cl_size)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.8 * z + 0.2 * w + u
    y = 1.0 + 1.15 * x + 0.35 * w + u
    lon_cl = rng.uniform(size=n_cluster)
    lat_cl = rng.uniform(size=n_cluster)
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "w": w,
            "z": z,
            "cl": cl,
            "coord_x": lon_cl[cl - 1],
            "coord_y": lat_cl[cl - 1],
        }
    )


def make_clustered_fe_iv_data(
    seed: int, n_fe: int = 15, t_per_fe: int = 6
) -> pd.DataFrame:
    """Build a clustered IV dataset with one-way absorbed fixed effects."""
    rng = np.random.default_rng(seed)
    n = n_fe * t_per_fe
    fe = np.repeat(np.arange(1, n_fe + 1), t_per_fe)
    cl = np.repeat(np.arange(1, n // 2 + 1), 2)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.75 * z + 0.25 * w + u
    y = 1.0 + 1.05 * x + 0.45 * w + rng.normal(size=n_fe)[fe - 1] + u
    coord_x_cl = rng.uniform(size=len(np.unique(cl)))
    coord_y_cl = rng.uniform(size=len(np.unique(cl)))
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "w": w,
            "z": z,
            "fe": fe,
            "cl": cl,
            "coord_x": coord_x_cl[cl - 1],
            "coord_y": coord_y_cl[cl - 1],
        }
    )


def run_r_script(
    script: str,
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute an R script from a temporary file for cross-platform stability."""
    assert R is not None, "Rscript not found."

    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".R",
            delete=False,
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(textwrap.dedent(script).strip())
            script_path = Path(handle.name)

        return subprocess.run(
            [R, "--vanilla", str(script_path)],
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        if script_path is not None:
            script_path.unlink(missing_ok=True)


def ensure_scpc_r_installed() -> None:
    """Install the reference scpcR package into the repo-local pytest cache."""
    assert R is not None, "Rscript not found."

    expected_files = [
        SCPC_R_LIBRARY / "scpcR" / "DESCRIPTION",
        SCPC_R_LIBRARY / "fixest" / "DESCRIPTION",
        SCPC_R_LIBRARY / "sandwich" / "DESCRIPTION",
    ]
    if all(path.exists() for path in expected_files) and not any(
        SCPC_R_LIBRARY.glob("00LOCK*")
    ):
        return

    if SCPC_R_LIBRARY.exists():
        shutil.rmtree(SCPC_R_LIBRARY)
    SCPC_R_LIBRARY.mkdir(parents=True, exist_ok=True)

    # install into the repo-local cache so parity tests do not depend on the
    # user's normal r library or a globally installed copy of scpcr.
    script = textwrap.dedent(
        f"""
        .libPaths(c({SCPC_R_LIBRARY.as_posix()!r}, .Library))
        repos <- c(CRAN = "https://cloud.r-project.org")
        options(repos = repos)

        needed <- c("remotes", "jsonlite")
        missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
        if (length(missing)) {{
          install.packages(missing, repos = repos, lib = .libPaths()[1], quiet = TRUE)
        }}

        remotes::install_url(
          "{SCPC_R_TARBALL_URL}",
          lib = .libPaths()[1],
          upgrade = "never",
          dependencies = TRUE,
          quiet = TRUE
        )
        remotes::install_version(
          "fixest",
          version = "{FIXEST_R_VERSION}",
          lib = .libPaths()[1],
          upgrade = "never",
          quiet = TRUE
        )
        remotes::install_version(
          "sandwich",
          version = "{SANDWICH_R_VERSION}",
          lib = .libPaths()[1],
          upgrade = "never",
          quiet = TRUE
        )
        """
    )

    # run one small r script that bootstraps remotes and then installs scpcr
    # directly into the cached test library for later reuse.
    completed = run_r_script(script)
    assert completed.returncode == 0, (
        completed.stderr.strip() or completed.stdout.strip()
    )


def execute_r_code(code: str, *, payload: Any | None = None) -> Any:
    """Execute R code against the cached scpcR install and return JSON."""
    assert R is not None, "Rscript not found."

    ensure_scpc_r_installed()
    stdin_text = "" if payload is None else json.dumps(payload)

    script = textwrap.dedent(
        f"""
        .libPaths(c({SCPC_R_LIBRARY.as_posix()!r}, .Library))
        suppressPackageStartupMessages(library(jsonlite))
        suppressPackageStartupMessages(library(scpcR))

        expected_versions <- list(
          scpcR = "{SCPC_R_VERSION}",
          fixest = "{FIXEST_R_VERSION}",
          sandwich = "{SANDWICH_R_VERSION}"
        )
        for (pkg in names(expected_versions)) {{
          actual <- as.character(
            utils::packageDescription(pkg, lib.loc = .libPaths()[1])$Version
          )
          if (!identical(actual, expected_versions[[pkg]])) {{
            stop(
              "Reference package version mismatch for ", pkg, ": expected ",
              expected_versions[[pkg]], ", got ", actual, "."
            )
          }}
        }}

        stdin_text <- paste(readLines(file("stdin"), warn = FALSE), collapse = "\\n")
        payload <- if (nzchar(stdin_text)) fromJSON(stdin_text, simplifyVector = FALSE) else NULL

        result <- NULL

        {code}

        cat(toJSON(result, auto_unbox = TRUE, digits = 17, null = "null"))
        """
    )

    # keep the python to r boundary very simple: json goes in on stdin and
    # json comes back on stdout.
    completed = run_r_script(script, input_text=stdin_text)
    assert completed.returncode == 0, (
        completed.stderr.strip() or completed.stdout.strip()
    )
    assert completed.stdout.strip(), (
        completed.stderr.strip() or "R subprocess returned no JSON output."
    )
    return json.loads(completed.stdout)
