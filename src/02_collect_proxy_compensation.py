"""
02_collect_proxy_compensation.py
Locate DEF 14A proxy filings on SEC EDGAR and extract CEO total compensation
from the Summary Compensation Table (SCT).

IMPORTANT: SCT structure is NOT standardized across firms. This script gets
you most of the way; expect to spot-check and hand-correct a residual set.
Manual overrides live in data/processed/manual_comp_entries.csv and take
precedence in 06_build_panel.py.

Output: data/processed/compensation_raw.csv
"""

import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

USER_AGENT = os.environ.get("SEC_USER_AGENT")
if not USER_AGENT:
    sys.exit("Set SEC_USER_AGENT env var, e.g. 'Jane Doe jane@example.com'")
HEADERS = {"User-Agent": USER_AGENT}
SLEEP = 0.15
YEARS = range(2019, 2025)


def get_filing_index(cik: str) -> pd.DataFrame:
    """All recent filings for a CIK from the submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    df = pd.DataFrame(recent)
    return df[df["form"] == "DEF 14A"].copy()


def filing_url(cik: str, accession: str, primary_doc: str) -> str:
    acc = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary_doc}"


def parse_money(text: str) -> float | None:
    """'$21,032,714' -> 21032714.0"""
    cleaned = re.sub(r"[^\d.]", "", text or "")
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def extract_sct_ceo_row(html: str) -> list[dict]:
    """
    Find tables near 'Summary Compensation Table' heading, take rows whose
    first cell looks like a named executive, return name / year / last money
    column (usually Total). Heuristic by design.
    """
    soup = BeautifulSoup(html, "lxml")
    results = []
    anchor = soup.find(string=re.compile(r"Summary\s+Compensation\s+Table", re.I))
    if not anchor:
        return results

    # search the next few tables after the heading
    node = anchor.parent
    tables_checked = 0
    while node and tables_checked < 3:
        node = node.find_next("table")
        if node is None:
            break
        tables_checked += 1
        for tr in node.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if len(cells) < 4:
                continue
            year_match = re.search(r"\b(20(1[9]|2[0-4]))\b", " ".join(cells[:3]))
            if not year_match:
                continue
            total = parse_money(cells[-1])
            if total and total > 100_000:  # sanity floor
                results.append(
                    {
                        "row_name": cells[0],
                        "sct_year": int(year_match.group(1)),
                        "total_comp": total,
                    }
                )
    return results


def main() -> None:
    financials = pd.read_csv(OUT / "financials.csv", dtype={"cik": str})
    firms = financials[["ticker", "cik", "company"]].drop_duplicates()

    rows = []
    for _, firm in tqdm(firms.iterrows(), total=len(firms)):
        try:
            filings = get_filing_index(firm["cik"])
        except requests.RequestException:
            print(f"  submissions fetch failed: {firm['ticker']}")
            continue
        time.sleep(SLEEP)

        for _, f in filings.iterrows():
            fdate = str(f["filingDate"])[:4]
            if not fdate.isdigit() or not (2019 <= int(fdate) <= 2025):
                continue
            url = filing_url(firm["cik"], f["accessionNumber"], f["primaryDocument"])
            try:
                html = requests.get(url, headers=HEADERS, timeout=60).text
            except requests.RequestException:
                continue
            time.sleep(SLEEP)

            for hit in extract_sct_ceo_row(html):
                rows.append(
                    {
                        "ticker": firm["ticker"],
                        "cik": firm["cik"],
                        "filing_date": f["filingDate"],
                        "filing_url": url,
                        **hit,
                    }
                )

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["ticker", "sct_year", "row_name"], keep="last"
    )
    df.to_csv(OUT / "compensation_raw.csv", index=False)
    print(f"Saved {len(df)} candidate SCT rows -> {OUT/'compensation_raw.csv'}")
    print("NEXT: review rows, identify CEO per firm-year, fill "
          "data/processed/manual_comp_entries.csv for parse failures.")


if __name__ == "__main__":
    main()
