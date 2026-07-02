# Data Sources & Collection Notes

## 1. SEC EDGAR XBRL API — firm financials ✅ automated

- Endpoint: `https://data.sec.gov/api/xbrl/companyfacts/CIK{10-digit}.json`
- Free, no key. **SEC requires a descriptive User-Agent** with contact email; anonymous requests are blocked. Set `SEC_USER_AGENT` env var.
- Rate limit: max 10 requests/second — scripts sleep 0.15s between calls.
- Ticker → CIK mapping: `https://www.sec.gov/files/company_tickers.json`
- Revenue reported under multiple us-gaap tags depending on firm/year. Priority order used:
  1. `RevenueFromContractWithCustomerExcludingAssessedTax`
  2. `Revenues`
  3. `SalesRevenueNet`
  Tag actually used is recorded per observation.

## 2. SEC DEF 14A proxy filings — CEO compensation ⚠️ semi-automated

- Filing index per company: `https://data.sec.gov/submissions/CIK{10-digit}.json` → filter `form == "DEF 14A"`.
- The Summary Compensation Table is an HTML table with **no standardized structure** across firms. `02_collect_proxy_compensation.py` locates candidate tables by keyword ("Summary Compensation Table") and extracts rows, but **output must be spot-checked**. Expect manual fixes for ~10–20% of firms.
- Alternative if parsing fails: hand-enter from proxy PDFs for the residual set (documented in `data/processed/manual_comp_entries.csv`).

## 3. CEO shareholder letters ⚠️ heterogeneous, partially manual

There is **no SEC form or API for shareholder letters**. Letters appear in:
- Annual report PDFs (ARS filings on EDGAR, form type `ARS`)
- 10-K wrap documents
- Company IR websites

`03_download_shareholder_letters.py` pulls ARS filings from EDGAR where available and extracts the letter section heuristically (first-person opening addressed "to shareholders"). Coverage is incomplete; missing letters collected manually and stored in `data/raw/letters/{ticker}_{year}.txt`. Provenance logged in `data/raw/letters/manifest.csv`.

## 4. Loughran–McDonald dictionary

- Download master dictionary from the LM website (Notre Dame Software Repository for Accounting and Finance). Not redistributed in this repo per their license — script expects it at `data/raw/LoughranMcDonald_MasterDictionary.csv`.

## 5. FinBERT

- Model: `ProsusAI/finbert` via Hugging Face `transformers`. Downloads ~440MB on first run; cached locally.
