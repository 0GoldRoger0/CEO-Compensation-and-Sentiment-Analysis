"""
08_xgboost_shap.py
XGBoost regression on log CEO compensation with grouped (by-firm) CV to
prevent leakage, then SHAP interpretation: does sentiment matter after
trees exhaust the financials?

Input:  data/processed/panel.csv
Output: results/xgb_metrics.txt, results/shap_summary.png,
        results/shap_dependence_finbert.png, results/feature_importance.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

BASE = Path(__file__).resolve().parents[1]
P = BASE / "data" / "processed"
R = BASE / "results"
R.mkdir(exist_ok=True)

FEATURES = [
    "finbert_net", "finbert_pos", "finbert_neg",
    "lm_net", "lm_uncertainty", "letter_word_count",
    "log_revenue", "roa",
]
TARGET = "log_total_comp"


def main() -> None:
    df = pd.read_csv(P / "panel.csv").dropna(subset=FEATURES + [TARGET])
    year_dummies = pd.get_dummies(df["fiscal_year"], prefix="fy", drop_first=True)
    X = pd.concat([df[FEATURES], year_dummies], axis=1)
    y = df[TARGET]
    groups = df["ticker"]

    # holdout split grouped by firm
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    # grouped 5-fold CV on training set
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    for tr, va in gkf.split(X_tr, y_tr, groups.iloc[tr_idx]):
        model.fit(X_tr.iloc[tr], y_tr.iloc[tr])
        cv_scores.append(r2_score(y_tr.iloc[va], model.predict(X_tr.iloc[va])))

    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = (
        f"Grouped 5-fold CV R2 (train): {np.mean(cv_scores):.3f} "
        f"(+/- {np.std(cv_scores):.3f})\n"
        f"Holdout R2:  {r2_score(y_te, pred):.3f}\n"
        f"Holdout MAE: {mean_absolute_error(y_te, pred):.3f} (log-dollars)\n"
        f"n_train={len(X_tr)}, n_test={len(X_te)}\n"
    )
    (R / "xgb_metrics.txt").write_text(metrics)
    print(metrics)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    plt.figure()
    shap.summary_plot(shap_values, X_te, show=False)
    plt.tight_layout()
    plt.savefig(R / "shap_summary.png", dpi=200)
    plt.close()

    plt.figure()
    shap.dependence_plot("finbert_net", shap_values, X_te,
                         interaction_index="log_revenue", show=False)
    plt.tight_layout()
    plt.savefig(R / "shap_dependence_finbert.png", dpi=200)
    plt.close()

    imp = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(R / "feature_importance.csv", index=False)
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
