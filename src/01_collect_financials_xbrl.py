"""
01_collect_financials_xbrl.py
Pull annual firm financials (revenue, net income, total assets) for S&P 500
firms from the SEC EDGAR XBRL companyfacts API, fiscal years 2019-2024.

Output: data/processed/financials.csv

SEC requires a descriptive User-Agent with contact info:
    export SEC_USER_AGENT="Your Name your.email@example.com"
Rate limit: <=10 req/sec. We sleep between calls.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

USER_AGENT = os.environ.get("SEC_USER_AGENT")
if not USER_AGENT:
    sys.exit("Set SEC_USER_AGENT env var, e.g. 'Jane Doe jane@example.com'")

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
YEARS = range(2019, 2025)
SLEEP = 0.15  # seconds between requests

# Revenue tags in priority order (firms differ)
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
CONCEPTS = {
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
}


def get_ticker_cik_map() -> pd.DataFrame:
    """SEC-maintained ticker -> CIK mapping."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).T
    df["cik"] = df["cik_str"].astype(int).astype(str).str.zfill(10)
    return df[["ticker", "cik", "title"]].rename(columns={"title": "company"})


def load_sp500_tickers() -> list[str]:
    """
    Expects data/raw/sp500_tickers.csv with a 'ticker' column.
    Build this once from a constituent list (e.g., Wikipedia S&P 500 table).
    """
    path = RAW / "sp500_tickers.csv"
    if not path.exists():
        sys.exit(f"Missing {path}. Create it with one column: ticker")
    return pd.read_csv(path)["ticker"].str.upper().str.strip().tolist()


def fetch_companyfacts(cik: str) -> dict | None:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


def extract_annual(facts: dict, tags: list[str]) -> tuple[dict, str | None]:
    """
    Return {fiscal_year: value} for the first tag that has 10-K (FY) data,
    plus the tag used. Prefers form 10-K, fp == 'FY'.
    """
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag not in usgaap:
            continue
        units = usgaap[tag].get("units", {}).get("USD", [])
        vals = {}
        for item in units:
            if item.get("form") == "10-K" and item.get("fp") == "FY":
                fy = item.get("fy")
                if fy in YEARS:
                    # later filings (amendments) overwrite earlier — keep latest
                    vals[fy] = item.get("val")
        if vals:
            return vals, tag
    return {}, None


def main() -> None:
    cik_map = get_ticker_cik_map()
    tickers = load_sp500_tickers()
    universe = cik_map[cik_map["ticker"].isin(tickers)]
    missing = set(tickers) - set(universe["ticker"])
    if missing:
        print(f"WARNING: no CIK found for {sorted(missing)}")

    rows = []
    for _, firm in tqdm(universe.iterrows(), total=len(universe)):
        facts = fetch_companyfacts(firm["cik"])
        time.sleep(SLEEP)
        if facts is None:
            print(f"  fetch failed: {firm['ticker']}")
            continue

        revenue, rev_tag = extract_annual(facts, REVENUE_TAGS)
        net_income, _ = extract_annual(facts, CONCEPTS["net_income"])
        assets, _ = extract_annual(facts, CONCEPTS["total_assets"])

        for fy in YEARS:
            rows.append(
                {
                    "ticker": firm["ticker"],
                    "cik": firm["cik"],
                    "company": firm["company"],
                    "fiscal_year": fy,
                    "revenue": revenue.get(fy),
                    "revenue_tag_used": rev_tag,
                    "net_income": net_income.get(fy),
                    "total_assets": assets.get(fy),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "financials.csv", index=False)
    filled = df["revenue"].notna().mean()
    print(f"Saved {len(df)} firm-years -> {OUT/'financials.csv'} "
          f"(revenue coverage {filled:.0%})")


if __name__ == "__main__":
    main()
