# Corrected SCPC reference results

Generated November 17, 2026 from the working scpcR source based on commit
`5ac0fd9ce02f189e3947ffd23776ed905f8082ce`, with the IV/FE and numerical
corrections applied, using the native backend and fixest 0.14.0. The source
SHA-256 values were `e4ad99ea2980f80e0da18ef3e91b9fd53068a743540ff14b8f4cf5d984c93076`
for `R/core.R`, `6cdff2c8b7c32dd2af388779bf0098f17b3ef7fecea276683dd53057f1656389`
for `R/utils-matrix.R`, and
`24d05e4db8844ec7837cff666bab9aec683c40d96cef2578f09bb00089aaee76`
for the vendored `scpc_op.rs`. The R expressions and inputs remain in the
associated Python tests; the JSON files record their outputs. The pinned
release oracle remains used for comparisons unaffected by these corrections.
Independent projection, calibration, and rejection-size checks supplement
these cross-language fixtures.

The corrected R sources are now committed in `scpcR@c6847e42e2f98764be08a8a4a29c2bb4485ef1f0`. The numerical source hashes above are unchanged.
