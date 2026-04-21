---
name: Bug report
about: Report a reproducible problem with scpc-python
title: "[bug] "
labels: bug
---

## Summary

Describe the problem in 1-3 sentences.

## Minimal reproduction

Provide a copy-pasteable example if possible.

```python

```

## Expected behavior

What did you expect to happen?

## Actual behavior

What happened instead? Include the full traceback, warning, or numerical mismatch if relevant.

```text

```

## Environment

- OS:
- Python version:
- install method: `uv` / `pip` / editable install / git install
- scpc-python version or commit:
- statsmodels version:
- affected function or workflow: `scpc` / `SCPCResult` / coordinates / clustering / docs / CI

## Additional context

Anything else that might help reproduce or explain the issue.

- Does this reproduce on synthetic data, example data, or only your own data?
- Does this affect `lon`/`lat`, `coords_euclidean`, clustering inputs, or result presentation?
- Is the issue a crash, wrong result, docs mismatch, install/import problem, or performance regression?
- If this is a numerical mismatch, do you see the same issue relative to the R package?
