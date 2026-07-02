"""
06_build_panel.py
Merge financials + compensation + both sentiment measures into the final
firm-year analysis panel. Applies manual compensation overrides, derives
log/ratio variables, documents dropped observations.

Inputs:  data/processed/financials.csv
         data/processed/compensation_raw.csv  (+ manual_comp_entries.csv)
         data/processed/sentiment_finbert.csv
         data/processed/sentiment_lm.csv
Output:  data/processed/panel.csv
         data/processed/dropped_observations.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
P = BASE / "data" / "processed"


def load_compensation() -> pd.DataFrame:
    """
    compensation_raw.csv has candidate SCT rows; a reviewed file
    compensation_reviewed.csv (ticker, fiscal_year, ceo_name, total_comp)
    is preferred if present. Manual entries override everything.
    """
    reviewed = P / "compensation_reviewed.csv"
    if reviewed.exists():
        comp = pd.read_csv(reviewed)
    else:
        raw = pd.read_csv(P / "compensation_raw.csv")
        # naive default: max total per firm-year (CEO usually highest paid).
        # This is a placeholder until manual review — flagged in output.
        comp = (
            raw.sort_values("total_comp", ascending=False)
            .drop_duplicates(subset=["ticker", "sct_year"])
            .rename(columns={"sct_year": "fiscal_year", "row_name": "ceo_name"})
            [["ticker", "fiscal_year", "ceo_name", "total_comp"]]
        )
        comp["comp_source"] = "auto_unreviewed"

    manual = P / "manual_comp_entries.csv"
    if manual.exists():
        m = pd.read_csv(manual)
        m["comp_source"] = "manual"
        comp = (
            pd.concat([comp, m], ignore_index=True)
            .sort_values("comp_source")  # manual last -> kept
            .drop_duplicates(subset=["ticker", "fiscal_year"], keep="last")
        )
    return comp


def main() -> None:
    fin = pd.read_csv(P / "financials.csv")
    comp = load_compensation()
    fb = pd.read_csv(P / "sentiment_finbert.csv")
    lm = pd.read_csv(P / "sentiment_lm.csv")

    panel = (
        fin.merge(comp, on=["ticker", "fiscal_year"], how="left")
           .merge(fb, on=["ticker", "fiscal_year"], how="left")
           .merge(lm, on=["ticker", "fiscal_year"], how="left")
    )

    # derived variables
    panel["log_total_comp"] = np.log(panel["total_comp"])
    panel["log_revenue"] = np.log(panel["revenue"])
    panel["roa"] = panel["net_income"] / panel["total_assets"]

    required = ["total_comp", "revenue", "net_income",
                "total_assets", "finbert_net", "lm_net"]
    complete = panel.dropna(subset=required)
    dropped = panel[~panel.index.isin(complete.index)].copy()
    dropped["drop_reason"] = dropped[required].isna().apply(
        lambda r: ",".join(r.index[r]), axis=1
    )

    complete.to_csv(P / "panel.csv", index=False)
    dropped.to_csv(P / "dropped_observations.csv", index=False)
    print(f"Panel: {len(complete)} firm-years "
          f"({complete['ticker'].nunique()} firms). "
          f"Dropped: {len(dropped)} (see dropped_observations.csv)")

    # quick validity check: do the two sentiment measures agree?
    corr = complete[["finbert_net", "lm_net"]].corr().iloc[0, 1]
    print(f"FinBERT vs LM net-sentiment correlation: {corr:.3f}")


if __name__ == "__main__":
    main()
