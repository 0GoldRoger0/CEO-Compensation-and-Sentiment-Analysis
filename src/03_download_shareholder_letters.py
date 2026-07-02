"""
03_download_shareholder_letters.py
Attempt automated collection of CEO shareholder letters from EDGAR ARS
(annual report to shareholders) filings. Letters have NO standardized SEC
form, so coverage is partial by nature — remaining letters are collected
manually from company IR sites and dropped into data/raw/letters/ as
{TICKER}_{YEAR}.txt, logged in manifest.csv.

Output: data/raw/letters/*.txt + data/raw/letters/manifest.csv
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
LETTERS = BASE / "data" / "raw" / "letters"
LETTERS.mkdir(parents=True, exist_ok=True)
OUT = BASE / "data" / "processed"

USER_AGENT = os.environ.get("SEC_USER_AGENT")
if not USER_AGENT:
    sys.exit("Set SEC_USER_AGENT env var, e.g. 'Jane Doe jane@example.com'")
HEADERS = {"User-Agent": USER_AGENT}
SLEEP = 0.15

LETTER_START = re.compile(
    r"(dear\s+(fellow\s+)?(share|stock)holders|to\s+our\s+(share|stock)holders)",
    re.I,
)


def get_ars_filings(cik: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["filings"]["recent"])
    return df[df["form"].isin(["ARS", "10-K"])].copy()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_letter(html: str, max_words: int = 4000) -> str | None:
    """
    Heuristic: find 'Dear Shareholders'-style opening, take text from there
    until a signature-like break or max_words. Returns None if no opening.
    """
    soup = BeautifulSoup(html, "lxml")
    full = clean_text(soup.get_text(" "))
    m = LETTER_START.search(full)
    if not m:
        return None
    snippet = full[m.start():]
    words = snippet.split()
    return " ".join(words[:max_words])


def main() -> None:
    financials = pd.read_csv(OUT / "financials.csv", dtype={"cik": str})
    firms = financials[["ticker", "cik"]].drop_duplicates()

    manifest = []
    for _, firm in tqdm(firms.iterrows(), total=len(firms)):
        try:
            filings = get_ars_filings(firm["cik"])
        except requests.RequestException:
            continue
        time.sleep(SLEEP)

        for _, f in filings.iterrows():
            year = str(f["filingDate"])[:4]
            if not (2019 <= int(year) <= 2025):
                continue
            dest = LETTERS / f"{firm['ticker']}_{year}.txt"
            if dest.exists():
                continue
            acc = f["accessionNumber"].replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/"
                   f"{int(firm['cik'])}/{acc}/{f['primaryDocument']}")
            try:
                html = requests.get(url, headers=HEADERS, timeout=60).text
            except requests.RequestException:
                continue
            time.sleep(SLEEP)

            letter = extract_letter(html)
            if letter and len(letter.split()) > 200:
                dest.write_text(letter, encoding="utf-8")
                manifest.append(
                    {"ticker": firm["ticker"], "year": year,
                     "source": "EDGAR", "url": url,
                     "words": len(letter.split())}
                )

    pd.DataFrame(manifest).to_csv(LETTERS / "manifest.csv", index=False)
    print(f"Auto-collected {len(manifest)} letters -> {LETTERS}")
    print("Add manually collected letters as TICKER_YEAR.txt and append "
          "rows to manifest.csv with source='manual'.")


if __name__ == "__main__":
    main()
