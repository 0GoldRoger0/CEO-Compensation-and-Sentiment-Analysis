"""
07_panel_regression.py
Two-way fixed effects panel regressions:
    log(comp_it) = b1*sentiment_it + controls + firm FE + year FE + e_it
Firm-clustered standard errors. Contemporaneous and t+1 lead specs.

Input:  data/processed/panel.csv
Output: results/regression_results.txt, results/regression_table.csv
"""

from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS

BASE = Path(__file__).resolve().parents[1]
P = BASE / "data" / "processed"
R = BASE / "results"
R.mkdir(exist_ok=True)

SENTIMENT_VARS = ["finbert_net", "lm_net", "lm_uncertainty"]
CONTROLS = ["log_revenue", "roa"]


def run_spec(df: pd.DataFrame, sentiment: str, dv: str) -> dict:
    data = df.dropna(subset=[dv, sentiment] + CONTROLS).copy()
    data = data.set_index(["ticker", "fiscal_year"])
    formula = f"{dv} ~ {sentiment} + {' + '.join(CONTROLS)} + EntityEffects + TimeEffects"
    res = PanelOLS.from_formula(formula, data=data).fit(
        cov_type="clustered", cluster_entity=True
    )
    return {
        "spec": f"{dv} ~ {sentiment}",
        "coef": res.params[sentiment],
        "se": res.std_errors[sentiment],
        "t": res.tstats[sentiment],
        "p": res.pvalues[sentiment],
        "n": int(res.nobs),
        "r2_within": res.rsquared_within,
        "full": res,
    }


def main() -> None:
    panel = pd.read_csv(P / "panel.csv")

    # lead DV: comp in t+1 matched to sentiment in t
    lead = panel[["ticker", "fiscal_year", "log_total_comp"]].copy()
    lead["fiscal_year"] -= 1
    lead = lead.rename(columns={"log_total_comp": "log_total_comp_lead"})
    panel = panel.merge(lead, on=["ticker", "fiscal_year"], how="left")

    results, texts = [], []
    for dv in ["log_total_comp", "log_total_comp_lead"]:
        for s in SENTIMENT_VARS:
            out = run_spec(panel, s, dv)
            results.append({k: v for k, v in out.items() if k != "full"})
            texts.append(f"===== {out['spec']} =====\n{out['full']}\n")

    pd.DataFrame(results).to_csv(R / "regression_table.csv", index=False)
    (R / "regression_results.txt").write_text("\n".join(texts))

    print(pd.DataFrame(results).round(4).to_string(index=False))
    print(f"\nFull output -> {R/'regression_results.txt'}")


if __name__ == "__main__":
    main()
