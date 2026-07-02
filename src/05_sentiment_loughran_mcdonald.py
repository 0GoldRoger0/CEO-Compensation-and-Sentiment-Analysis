"""
05_sentiment_loughran_mcdonald.py
Dictionary-based sentiment using the Loughran-McDonald master dictionary.

Requires: data/raw/LoughranMcDonald_MasterDictionary.csv
Download from the Notre Dame SRAF site (not redistributed in this repo).
Expected columns include: Word, Positive, Negative, Uncertainty, Litigious.

Output: data/processed/sentiment_lm.csv
"""

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
LETTERS = BASE / "data" / "raw" / "letters"
DICT_PATH = BASE / "data" / "raw" / "LoughranMcDonald_MasterDictionary.csv"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["Positive", "Negative", "Uncertainty", "Litigious"]


def load_lm_dictionary() -> dict[str, set[str]]:
    if not DICT_PATH.exists():
        raise SystemExit(
            f"Missing {DICT_PATH}. Download the LM master dictionary from "
            "the Notre Dame SRAF site and place it there."
        )
    lm = pd.read_csv(DICT_PATH)
    lm["Word"] = lm["Word"].astype(str).str.upper()
    return {
        cat: set(lm.loc[lm[cat] != 0, "Word"]) for cat in CATEGORIES
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]{2,}", text.upper())


def main() -> None:
    word_sets = load_lm_dictionary()
    rows = []
    files = sorted(LETTERS.glob("*_*.txt"))
    if not files:
        raise SystemExit(f"No letters found in {LETTERS}. Run 03 first.")

    for path in tqdm(files):
        ticker, year = path.stem.rsplit("_", 1)
        tokens = tokenize(path.read_text(encoding="utf-8"))
        n = len(tokens)
        if n < 100:
            continue

        shares = {
            f"lm_{cat.lower()}": sum(t in word_sets[cat] for t in tokens) / n
            for cat in CATEGORIES
        }
        rows.append(
            {
                "ticker": ticker,
                "fiscal_year": int(year),
                "letter_word_count": n,
                **shares,
                "lm_net": shares["lm_positive"] - shares["lm_negative"],
            }
        )

    df = pd.DataFrame(rows).rename(
        columns={"lm_positive": "lm_pos", "lm_negative": "lm_neg"}
    )
    df.to_csv(OUT / "sentiment_lm.csv", index=False)
    print(f"Scored {len(df)} letters -> {OUT/'sentiment_lm.csv'}")


if __name__ == "__main__":
    main()
