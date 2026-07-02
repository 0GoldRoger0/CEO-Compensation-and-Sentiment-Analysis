# Data Dictionary

## Folder policy

- `raw/` — downloaded SEC filings and letter texts. **Gitignored.** Reproduce with `src/01`–`03`.
- `processed/` — derived firm-year datasets committed to the repo (numbers only, no filing text).

## processed/panel.csv (final analysis dataset)

One row = one firm-year.

| Column | Type | Description | Source |
|---|---|---|---|
| `ticker` | str | Ticker symbol | S&P 500 constituent list |
| `cik` | str | SEC Central Index Key (10-digit, zero-padded) | SEC company_tickers.json |
| `company` | str | Company name | SEC |
| `fiscal_year` | int | Fiscal year (2019–2024) | — |
| `ceo_name` | str | CEO named in Summary Compensation Table | DEF 14A |
| `total_comp` | float | CEO total compensation, USD | DEF 14A Summary Compensation Table |
| `log_total_comp` | float | ln(total_comp) — regression DV | derived |
| `revenue` | float | Total revenue, USD | XBRL `Revenues` / `RevenueFromContractWithCustomerExcludingAssessedTax` |
| `net_income` | float | Net income, USD | XBRL `NetIncomeLoss` |
| `total_assets` | float | Total assets, USD | XBRL `Assets` |
| `log_revenue` | float | ln(revenue) — size control | derived |
| `roa` | float | net_income / total_assets | derived |
| `finbert_pos` | float | Share of sentences classified positive | FinBERT |
| `finbert_neg` | float | Share of sentences classified negative | FinBERT |
| `finbert_net` | float | finbert_pos − finbert_neg (primary sentiment measure) | derived |
| `lm_pos` | float | LM positive words / total words | Loughran–McDonald |
| `lm_neg` | float | LM negative words / total words | Loughran–McDonald |
| `lm_uncertainty` | float | LM uncertainty words / total words | Loughran–McDonald |
| `lm_net` | float | lm_pos − lm_neg | derived |
| `letter_word_count` | int | Cleaned letter length | derived |

## Known gaps / cleaning notes

- Some firms report revenue under multiple XBRL tags; `01_collect_financials_xbrl.py` tries tags in priority order and flags fills in `revenue_tag_used`.
- Fiscal years ≠ calendar years for some firms; matched on fiscal year end within the filing.
- Firms with no separately published CEO letter for a year are dropped (documented in `processed/dropped_observations.csv`).
