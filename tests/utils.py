from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCPC_R_GITHUB = "spatial-spur/scpcR"
SCPC_R_REF = "v0.1.0b1"
SCPC_R_LIBRARY = PROJECT_ROOT / ".pytest_cache" / "scpcR" / SCPC_R_REF / "lib"
R: str | None = shutil.which("Rscript")


def ensure_scpc_r_installed() -> None:
    """Install the reference scpcR package into the repo-local pytest cache."""
    assert R is not None, "Rscript not found."

    # nothing to do if its already installed
    if (SCPC_R_LIBRARY / "scpcR" / "DESCRIPTION").exists():
        return

    SCPC_R_LIBRARY.mkdir(parents=True, exist_ok=True)

    # install into the repo-local cache so parity tests do not depend on the
    # user's normal r library or a globally installed copy of scpcr.
    script = textwrap.dedent(
        f"""
        .libPaths(c({SCPC_R_LIBRARY.as_posix()!r}, .Library))
        repos <- c(CRAN = "https://cloud.r-project.org")

        needed <- c("remotes", "jsonlite")
        missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
        if (length(missing)) {{
          install.packages(missing, repos = repos, lib = .libPaths()[1], quiet = TRUE)
        }}

        remotes::install_github(
          "{SCPC_R_GITHUB}@{SCPC_R_REF}",
          lib = .libPaths()[1],
          upgrade = "never",
          dependencies = TRUE,
          quiet = TRUE
        )
        """
    ).strip()

    # run one small r script that bootstraps remotes and then installs scpcr
    # directly into the cached test library for later reuse.
    completed = subprocess.run(
        [R, "--vanilla", "-e", script],
        text=True,
        capture_output=True,
        check=False,
    )
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

        stdin_text <- paste(readLines(file("stdin"), warn = FALSE), collapse = "\\n")
        payload <- if (nzchar(stdin_text)) fromJSON(stdin_text, simplifyVector = FALSE) else NULL

        result <- NULL

        {code}

        cat(toJSON(result, auto_unbox = TRUE, digits = 17, null = "null"))
        """
    ).strip()

    # keep the python to r boundary very simple: json goes in on stdin and
    # json comes back on stdout.
    completed = subprocess.run(
        [R, "--vanilla", "-e", script],
        input=stdin_text,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stderr.strip() or completed.stdout.strip()
    )
    return json.loads(completed.stdout)
