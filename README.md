# Auditing and Explaining Diabetes Classification Models

**DTU 02820 — Explainable AI and User Experience · Assignment 1 · Group 10**

We compare a shallow decision tree and a small neural network on the balanced BRFSS 2015 health-indicator dataset. LIME and Kernel SHAP explain the same held-out respondent, while global SHAP summarizes 40 test respondents. Presentations adapt the findings for a head of data science, a patient-organization director and a patient.

The task is classification of **reported prediabetes or diabetes status**, not prediction of future disease onset. Scores and explanations are research outputs, not diagnoses, causal effects or estimates of the benefit of a lifestyle change.

## Start here

| File | Purpose |
| --- | --- |
| [model_decision_tree.ipynb](model_decision_tree.ipynb) | Training, evaluation, exact tree path, LIME and Kernel SHAP |
| [model_neural_network.ipynb](model_neural_network.ipynb) | Training, evaluation, LIME, 20-run LIME stability analysis and Kernel SHAP |
| [PowerPoint/](PowerPoint/) | Original role-play presentations for all three stakeholders |
| [requirements.txt](requirements.txt) | Compatible dependency ranges |
| [requirements-tested.txt](requirements-tested.txt) | Exact direct dependency versions used in the validation run |
| [VALIDATION.md](VALIDATION.md) | Execution checks and differences from the report |
| [scripts/check_submission.py](scripts/check_submission.py) | Notebook syntax, dataset, shared split and patient checks |
| [archive/](archive/) | Preserved notebooks from before submission cleanup, including exploratory PFI/IG |
| `old/`, `ignore folder, just some .md/`, `GAME PLAN.jpg` | Historical experiments and planning material; not the main report workflow |

The assignment specification is included in the repository. The final written report is submitted separately through DTU Learn. Original slides are retained as presentation evidence; they can include earlier exploratory methods that are not part of the final two-method analysis.

## Setup and execution

Use Python 3.12 in a separate environment (the validation run used Python 3.12.14). From the repository root:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install and launch:

```bash
python -m pip install -r requirements-tested.txt
python scripts/check_submission.py
python -m jupyterlab
```

Select the environment's kernel and **Restart Kernel and Run All Cells**. Run the decision-tree notebook first, then the neural-network notebook; each is independent and can also run alone. Keep the CSV next to the notebooks and launch Jupyter from the repository root. No absolute Windows paths are required. A GPU is optional; the NN selects CUDA when available. Kernel SHAP and repeated LIME runs are the slowest sections.

The existing CSV is `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`, the course-provided balanced BRFSS 2015 dataset. It has 70,692 rows, 21 input features and target `Diabetes_binary`. Preserve its row order: index 57608 is a zero-based row index, not a patient ID. If the CSV is missing, obtain that exact file from DTU Learn.

## Models and common reference person

- **Shared test set:** stratified 80/20 split, `random_state=42`; 14,139 test respondents.
- **Decision tree:** Gini criterion, maximum depth 4, minimum 100 samples per leaf; fitted on 56,553 respondents.
- **Neural network:** 21 → 32 ReLU → 1 logit (737 parameters), sigmoid score; Adam, learning rate 0.001, weight decay 0.0001, batch size 256, 30 epochs. Fitting uses 45,242 respondents and validation 11,311.
- **Our patient:** test respondent 57608, woman aged 65–69, BMI 53, fair general health, reported high blood pressure and high cholesterol, positive survey label. Both notebooks assert that she is in the test set and outside the training set, and display all raw inputs.
- **LIME:** 3,000 perturbations, ten displayed feature conditions, weighted Ridge surrogate; binary inputs categorical. NN stability uses 20 random seeds.
- **Kernel SHAP:** 50 training background rows, 40 global test rows, 200 coalition samples per explanation; patient explained separately. Positive-class contributions use the identity link.

## Report reference results

These are the recorded report results, **not guarantees for a new run**.

| Metric | Decision tree | Neural network |
| --- | ---: | ---: |
| Accuracy | 72.68% | 75.21% |
| ROC-AUC | 0.795 | 0.830 |
| False negatives | 1,574 | 1,397 |
| False positives | 2,289 | 2,108 |
| Patient score | 0.8526 | 0.8854 |
| Patient LIME weighted R² | 0.450 | 0.479 |

For the NN, repeated LIME explanations had mean weighted R² 0.4654 (SD 0.0122) and mean absolute local error 0.1231 (12.31 score percentage points). Fidelity describes surrogate agreement, not classifier accuracy. Stable feature selection does not guarantee faithful explanations.

## Reproducibility and exports

Main notebooks retain original saved outputs for the report, except the global SHAP cells, whose outputs were cleared when an explicit coalition-sampling seed was added. A fresh run may change global SHAP rankings. Model-fitting settings and the original preprocessing order have not been changed. Dependency ranges are not a reconstruction of the unknown original environment; software versions and CPU/GPU differences may affect scores and explanations. Compare fresh results with the report before replacing figures.

The last cell exports current-run evidence into `results/dt/` or `results/nn/`:

- `test_predictions.csv`: original dataset index, actual label, score and prediction.
- `patient_raw.csv`, `patient_lime.csv`, `patient_shap.csv`: the selected respondent and explanations.
- `global_shap.csv`: sorted mean absolute SHAP contributions.
- `run_metadata.json`: dataset SHA-256, package versions, seed, metrics and SHAP reconstruction error.
- NN additionally exports `lime_runs.csv` and `lime_stability.csv`.

These generated files are ignored by Git. To compare models statistically, join predictions by `dataset_index` and verify identical labels. No McNemar test was performed for the report; individual confidence intervals are not a paired test of model differences.

## Limitations relevant to the assignment

- The NN's original preprocessing is fitted on the outer training portion **before** its internal validation split. Validation rows therefore influence scaling/imputation statistics. Test rows remain separate. A future corrected experiment should split before fitting preprocessing and report the resulting new metrics.
- Training is fixed at 30 epochs, with no early stopping. The recorded curve contains validation loss only; it does not by itself establish overfitting.
- A balanced historical US sample is not representative of population prevalence; scores and precision cannot automatically be transferred to deployment.
- LIME fidelity is limited and some surrogate outputs exceed one. SHAP additivity does not independently validate individual contributions. Both methods can evaluate implausible combinations of dependent features.
- Global SHAP uses only 40 respondents. Stability across background samples and sampling budgets was not systematically tested.
- Error cases are identified, but detailed local error-case explanations and subgroup fairness evaluation remain future work. Role-play feedback is not a clinical or patient-usability validation.

## Authors and sources

Magnus Lau Ovesen (s245689), Rasmus Larsen (s256261), Victor Birkefeldt Carstens (s256191), Louise Seedorff (s246273).

Dataset: course-provided balanced BRFSS 2015 extract, DTU Learn; consult the accompanying BRFSS codebook for coding and interpretation. XAI background: [Molnar — LIME](https://christophm.github.io/interpretable-ml-book/lime.html) and [Molnar — SHAP](https://christophm.github.io/interpretable-ml-book/shap.html).
