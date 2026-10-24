# Submission validation — 24 October 2026

Both main notebooks were executed cell by cell, in order, in fresh independent Python namespaces with the repository CSV. Plotting used a non-interactive backend; rich notebook table display was suppressed for the execution check. The original notebook outputs were not overwritten. This is a full code execution check, not a browser/Jupyter UI test or visual review of each regenerated chart.

## Passed

- Valid notebook structure (nbformat) and compilable Python cells; no saved error outputs.
- Dataset shape 70,692 × 22, balanced target and no missing values.
- Identical ordered held-out indices and labels in both exported prediction files.
- Reference respondent 57608 belongs to the test set, not the training set; recorded patient values match the report.
- Decision-tree and NN training, evaluation, LIME, global/local Kernel SHAP and all export cells completed.
- All 20 NN LIME repetitions completed.
- Local SHAP reconstruction error was zero at the printed numerical precision for both models. This checks additivity only, not attribution validity.

## New run versus original report

| Quantity | Report | Validation run |
| --- | ---: | ---: |
| DT accuracy | 72.68% | 72.6784% |
| NN accuracy | 75.21% | 75.2528% |
| DT patient score | 0.8526 | 0.852626 |
| NN patient score | 0.8854 | 0.885580 |
| DT LIME weighted R² | 0.450 | 0.450322 |
| NN LIME weighted R² | 0.479 | 0.478345 |

The DT reference result is reproduced to reporting precision. The NN differs slightly in this environment. Do not claim exact reproduction or silently replace report values with this run. The report's saved outputs remain in the notebooks and the historical copies are preserved in `archive/`.

Python 3.12.14, Linux CPU; exact direct dependencies are in `requirements-tested.txt`. Execution used one CPU thread for OpenMP/MKL/OpenBLAS. The original software environment and model weights were not supplied as a lockfile/checkpoint.

Dataset SHA-256: `873128ba0264d66a8b09bb6158f21fd4bedc10459d1ed5ecad3b94167a9539b4`.

## Deliberately not changed

The original NN preprocessing order, model hyperparameters, training/test split and reference respondent remain unchanged. The validation preprocessing limitation is now documented. Global SHAP now receives an explicit random seed; its old outputs were cleared because fresh coalition sampling can change the ranking. No significance, subgroup-fairness or clinical-readiness claim is added.
