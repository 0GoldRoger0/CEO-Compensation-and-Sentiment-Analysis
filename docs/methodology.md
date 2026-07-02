# Methodology

## 1. Sample

S&P 500 constituents as of study start, fiscal years 2019–2024 (≈3,000 potential firm-years). Financial firms retained but flagged (`is_financial`) for robustness checks since compensation structures differ.

## 2. Variables

**Dependent variable:** `log_total_comp` — natural log of CEO total compensation from the Summary Compensation Table (salary + bonus + stock awards + option awards + non-equity incentive + pension change + other).

**Sentiment measures (independent variables of interest):**
- `finbert_net` — FinBERT sentence-level classification aggregated per letter: (positive sentences − negative sentences) / total sentences.
- `lm_net` — Loughran–McDonald dictionary: (positive word share − negative word share).
- `lm_uncertainty` — LM uncertainty word share (secondary hypothesis: hedged language ↔ pay).

Two methods used deliberately: FinBERT captures context ("did not decline" = positive); LM is transparent, replicable, and the finance-literature standard. Agreement between them is reported as a validity check.

**Controls:** `log_revenue` (firm size), `roa` (profitability), year fixed effects, firm fixed effects.

## 3. Econometric model

Two-way fixed effects panel:

```
log(comp_it) = β1·sentiment_it + β2·log(revenue_it) + β3·roa_it + α_i + γ_t + ε_it
```

- `α_i` firm FE absorbs time-invariant firm/CEO factors (industry, pay culture).
- `γ_t` year FE absorbs macro shocks (COVID 2020, 2022 market drawdown).
- Standard errors clustered by firm.
- Estimated with `linearmodels.PanelOLS`.

**Timing specifications:** contemporaneous (letter year t → comp year t) and lead (letter year t → comp year t+1), since letters may reflect pay already set or influence next cycle.

## 4. Machine learning layer

XGBoost regression on `log_total_comp` using sentiment features + financial controls + year dummies:
- Train/test split by firm (grouped) to prevent leakage across years of the same firm.
- Hyperparameters via 5-fold grouped CV.
- **SHAP** for global feature importance and dependence plots — the ML step is interpretive, not just predictive: it asks whether sentiment matters after trees have exhausted the financials.

## 5. Robustness

- Drop financials / drop 2020 (COVID year).
- Winsorize compensation at 1%/99%.
- Alternative DV: log(salary + bonus) cash-only compensation.
- Letter length as additional control (verbosity ≠ sentiment).

## 6. Limitations

- Letters are voluntary disclosures; selection into publishing a letter is nonrandom.
- Comp committees set pay with far more information than letter tone; results are associations.
- FinBERT was trained on analyst/news text, not letters specifically; domain shift possible.
