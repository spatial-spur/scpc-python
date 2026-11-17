# Historical scpcR v0.1.3 reference

These are archived outputs from the release oracle (scpcR v0.1.3, fixest
0.14.0) for the same expressions and inputs as the corrected reference files.
They document intentional changes: exact decay calibration, sampled large-n
geometry, critical-value search, and the IV/absorbed-FE projection. They are
not used as correctness targets for those changed calculations. Unaffected
comparisons elsewhere in the suite still call the pinned release directly.

Representative release values include decay `7.06090398799841` (corrected
`7.061305747643886`), clustered FE-IV slope p-value `0.013581584217679871`
(corrected `0.006275741868426211`), and one-way FE-IV slope p-value
`0.009807065049653708` (corrected `0.0047588271202008495`). The independent
matrix and rejection-probability tests in `test_projection_corrections.py`
check the corrected projection; calibration tests check its defining equation.
