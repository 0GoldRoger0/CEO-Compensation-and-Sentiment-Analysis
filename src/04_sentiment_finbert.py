"""
04_sentiment_finbert.py
Score each shareholder letter with FinBERT (ProsusAI/finbert).
Sentence-level classification -> letter-level aggregate shares.

Output: data/processed/sentiment_finbert.csv
Note: first run downloads ~440MB model. GPU used if available.
"""

import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
LETTERS = BASE / "data" / "raw" / "letters"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]  # ProsusAI/finbert label order
BATCH = 16
MAX_LEN = 128


def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.split()) >= 4]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    model.eval()

    rows = []
    files = sorted(LETTERS.glob("*_*.txt"))
    if not files:
        raise SystemExit(f"No letters found in {LETTERS}. Run 03 first.")

    for path in tqdm(files):
        ticker, year = path.stem.rsplit("_", 1)
        sents = split_sentences(path.read_text(encoding="utf-8"))
        if not sents:
            continue

        counts = {l: 0 for l in LABELS}
        with torch.no_grad():
            for i in range(0, len(sents), BATCH):
                batch = sents[i:i + BATCH]
                enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                                padding=True, return_tensors="pt").to(device)
                preds = model(**enc).logits.argmax(dim=1).tolist()
                for p in preds:
                    counts[LABELS[p]] += 1

        n = len(sents)
        rows.append(
            {
                "ticker": ticker,
                "fiscal_year": int(year),
                "n_sentences": n,
                "finbert_pos": counts["positive"] / n,
                "finbert_neg": counts["negative"] / n,
                "finbert_neu": counts["neutral"] / n,
                "finbert_net": (counts["positive"] - counts["negative"]) / n,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sentiment_finbert.csv", index=False)
    print(f"Scored {len(df)} letters -> {OUT/'sentiment_finbert.csv'}")


if __name__ == "__main__":
    main()
