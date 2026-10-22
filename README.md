# BRFSS 2023 — Diabetes Classification (replication of Muhammad et al., 2025)

Replication of the machine-learning pipeline from:

> Muhammad, M. A., Sani, J., & Ahmed, M. M. (2025). *Exploring Explainable Machine Learning for
> Predicting and Interpreting Self-Reported Diabetes among Tennessee Adults: Insights from the 2023
> Behavioral Risk Factor Surveillance System (BRFSS).* Journal of Primary Care & Community Health, 16.
> https://doi.org/10.1177/21501319251400546

## What is replicated

- 80:20 stratified split, `random_state=42`
- Imputation + one-hot encoding fit on the training set only
- SMOTE applied to the training set only (test set never resampled)
- Seven models: Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- Seven metrics + ROC curve + SHAP TreeExplainer

## What is NOT replicated

| Paper | This repo | Why |
|---|---|---|
| Tennessee subset, n = 5,634 | Stratified **national** sample, n = 5,634 | The re-downloadable BRFSS extract does not include `_STATE`, so the Tennessee filter is impossible from it. Sample size matched to the paper's cohort scale. |
| Grid search, 5-fold CV | Default hyperparameters, single holdout | Not part of the replication target |
| Weighted survey estimates | Unweighted | Paper's ML also runs unweighted |

This is a **pipeline replication**, not a numeric replication. Do not cite these metrics as a
reproduction of the paper's Tennessee results.

## Results (holdout 20%, 821 positives / 4,813 negatives)

| Model | Accuracy | Precision | Recall | F1 | Balanced Acc | AUROC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8438 | 0.4444 | 0.2927 | 0.3529 | 0.6152 | **0.7876** | **0.4000** |
| Random Forest | 0.8598 | 0.5577 | 0.1768 | 0.2685 | 0.5765 | 0.7835 | 0.3954 |
| XGBoost | 0.8527 | 0.4875 | 0.2378 | 0.3197 | 0.5976 | 0.7777 | 0.3991 |
| Logistic Regression | 0.7090 | 0.2887 | **0.6829** | 0.4058 | **0.6982** | 0.7731 | 0.3689 |
| SVM | 0.8270 | 0.3946 | 0.3537 | 0.3730 | 0.6306 | 0.7726 | 0.3887 |
| KNN | 0.5475 | 0.2136 | 0.7866 | 0.3359 | 0.6467 | 0.6808 | 0.2245 |
| Decision Tree | 0.7782 | 0.2500 | 0.2622 | 0.2560 | 0.5641 | 0.5682 | 0.1731 |

Gradient Boosting leads on AUROC and PR-AUC but **not** on recall. For a screening framing, that
trade-off is the result — "best model" is metric-dependent.

## SHAP (Gradient Boosting, top features)

1. `BPHIGH6_1.0` — hypertension history, widest spread
2. `ALCDAY4_201.0` — alcohol-day code; **encoding artifact**, see Limitations
3. `GENHLTH_1.0` — "Excellent" general health, pushes predictions negative
4. `_BMI5CAT_4.0` — obesity category
5. `GENHLTH_2.0`, `MENTHLTH_30.0`, `_AGEG5YR_11.0`, `GENHLTH_4.0`, `_AGEG5YR_12.0`

## Limitations

> **Recode bug in the replication notebook (found 2026-10-22).** BRFSS uses `888` (ALCDAY4) and
> `88` (MENTHLTH/PHYSHLTH) for "none". The replication notebook treated them as missing and
> mode-imputed them, so 44% of non-drinkers became `ALCDAY4 = 201` and 59% of respondents with zero
> poor-mental-health days became `MENTHLTH = 30`. AUROC is unaffected; **the SHAP interpretation of
> those two features in the replication is invalid.** Use feature set C in the ablation below.

1. **Cohort is not Tennessee.** National stratified proxy at the paper's sample size.
2. **No hyperparameter tuning, no cross-validation.** Deviates from the paper's grid search.
3. **Categorical encoding of numeric BRFSS codes.** `ALCDAY4`, `MENTHLTH`, `PHYSHLTH` were one-hot
   encoded as if nominal, which creates levels like `ALCDAY4_201.0`. That level ranking second in
   SHAP is an artifact of the encoding, not a clinical finding.
4. **Outcome definition follows the paper**, which counts `DIABETE4 = 2` (gestational diabetes only)
   as positive. Clinically debatable.
5. **Unweighted.** Results do not represent the US adult population.
6. **Single holdout, no confidence intervals.** Metric differences under ~0.02 AUROC are not
   distinguishable at this test-set size.

## Ablation: post-diagnosis leakage and encoding

`ablation_brfss2023_leakage_encoding.ipynb` → `outputs_ablation/`. Same cohort, split, and SMOTE.

| Set | Change | Features | GB AUROC (holdout) | GB AUROC (5-fold CV) |
|---|---|---:|---:|---:|
| A_paper | replication | 183 | 0.7808 | 0.7871 ± 0.0243 |
| B_no_leakage | A minus 11 post-diagnosis proxies (Noh et al., 2026) | 161 | 0.7817 | 0.7714 ± 0.0245 |
| C_ordinal | B with ALCDAY4/MENTHLTH/PHYSHLTH as days/month, 888/88 → 0 | 76 | **0.7877** | 0.7852 ± 0.0240 |

- 21 DeLong tests (7 models × 3 pairs), Bonferroni-corrected: **no significant difference**.
  GB A→B: Δ = −0.0009, p = 0.92.
- Removing post-diagnosis proxies does not reduce discrimination; the model is not driven by leakage.
- Set C reaches the same AUROC with 76 instead of 183 features, and its SHAP ranking is clinically
  readable: cholesterol, general health, obesity, age group, alcohol days/month.
- Logistic Regression on C: AUROC 0.7865, recall 0.7256 — the better screening candidate.

## Files

- `replikasi_brfss2023_diabetes_muhammad2025.ipynb` — executed notebook, 12 cells
- `outputs/model_metrics.csv` — metrics for all seven models
- `outputs/model_comparison_and_roc.png` — metric bar plot + ROC curves
- `outputs/shap_beeswarm.png` — SHAP beeswarm, Gradient Boosting
- `outputs/reproducibility.json` — pipeline metadata

## Reproduce

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  replikasi_brfss2023_diabetes_muhammad2025.ipynb
```

The notebook downloads the BRFSS 2023 extract on first run via the BRFSS Data Explorer endpoint
(`POST https://brfssexplore.org/download`). The CDC `LLCP2023XPT.zip` endpoint returns HTTP 403 to
non-browser clients. Expect ~433k rows and 30 columns before the outcome filter.

Note: `shap` imports `torch` opportunistically on Windows and fails if the local PyTorch install is
broken. The notebook stubs `sys.modules['torch']` before importing `shap`; TreeExplainer does not
need it.

## Data source

Centers for Disease Control and Prevention. *Behavioral Risk Factor Surveillance System Survey Data*,
2023. U.S. Department of Health and Human Services, Atlanta, Georgia.

## License

Code: MIT. Derived data and figures: CC BY 4.0. BRFSS data itself is public domain.
