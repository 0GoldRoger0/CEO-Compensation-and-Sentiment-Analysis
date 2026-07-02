# Does What CEOs Say Predict What They Earn?

**NLP sentiment analysis of CEO shareholder letters vs. executive compensation — S&P 500, 2019–2024**

MS Applied Data Analytics Capstone (DATA 5130) · University of Detroit Mercy · Advisor: Dr. Yu Peng Lin

---

## Research Question

Does the tone and sentiment of a CEO's annual shareholder letter predict — or reflect — how that CEO is paid? This project quantifies the language of CEO communication using financial-domain NLP and tests its relationship with executive compensation, controlling for firm performance and size.

**Hypotheses:**
- **H1:** More positive/optimistic letter sentiment is associated with higher total CEO compensation in the same or following fiscal year.
- **H2:** Sentiment adds explanatory power for compensation beyond standard financial controls (revenue, net income, market cap, TSR).
- **H3:** Nonlinear ML models (XGBoost) capture sentiment–pay interactions that linear panel models miss.

## Data

| Source | What | How |
|---|---|---|
| SEC EDGAR XBRL API (`data.sec.gov`) | Firm financials (revenue, net income, assets) | `src/01_collect_financials_xbrl.py` |
| SEC DEF 14A proxy filings | CEO total compensation (Summary Compensation Table) | `src/02_collect_proxy_compensation.py` |
| Annual reports / 10-K exhibits / IR sites | CEO shareholder letters (text) | `src/03_download_shareholder_letters.py` |

**Scope:** S&P 500 constituents, fiscal years 2019–2024.

> **Note on data availability:** Raw filings and letter texts are **not committed** to this repo (size + copyright). The scripts reproduce the collection pipeline. Processed panel datasets (firm-year level, derived metrics only) are in `data/processed/` where redistribution is permissible. See `data/README.md`.

## Methodology

1. **Text collection & cleaning** — extract CEO letters, strip boilerplate, normalize.
2. **Sentiment scoring (two methods, cross-validated against each other):**
   - **FinBERT** (`ProsusAI/finbert`) — transformer fine-tuned on financial text; sentence-level positive/negative/neutral, aggregated per letter.
   - **Loughran–McDonald dictionary** — finance-specific word lists (positive, negative, uncertainty, litigious); proportion-based scores.
3. **Panel construction** — firm-year panel joining sentiment, compensation, and financial controls.
4. **Panel regression** — fixed effects (firm + year), clustered standard errors. Dependent variable: log(total CEO compensation).
5. **XGBoost + SHAP** — gradient-boosted trees to capture nonlinearity; SHAP values for interpretability and feature importance.

Full detail: [`docs/methodology.md`](docs/methodology.md)

## Repository Structure

```
├── data/
│   ├── raw/                  # downloaded filings (gitignored)
│   ├── processed/            # cleaned panel datasets
│   └── README.md             # data dictionary
├── docs/
│   ├── methodology.md        # full methodology write-up
│   └── data_sources.md       # source details, API notes, limitations
├── notebooks/                # exploratory analysis notebooks
├── results/                  # regression tables, SHAP plots, figures
├── src/
│   ├── 01_collect_financials_xbrl.py
│   ├── 02_collect_proxy_compensation.py
│   ├── 03_download_shareholder_letters.py
│   ├── 04_sentiment_finbert.py
│   ├── 05_sentiment_loughran_mcdonald.py
│   ├── 06_build_panel.py
│   ├── 07_panel_regression.py
│   └── 08_xgboost_shap.py
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/0GoldRoger0/ceo-sentiment-compensation.git
cd ceo-sentiment-compensation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set your SEC-required identity (SEC blocks anonymous requests)
export SEC_USER_AGENT="Your Name your.email@example.com"

# Run pipeline in order
python src/01_collect_financials_xbrl.py
python src/02_collect_proxy_compensation.py
python src/03_download_shareholder_letters.py
python src/04_sentiment_finbert.py
python src/05_sentiment_loughran_mcdonald.py
python src/06_build_panel.py
python src/07_panel_regression.py
python src/08_xgboost_shap.py
```

## Tech Stack

Python · pandas · requests · BeautifulSoup · transformers (FinBERT) · statsmodels / linearmodels (panel FE) · XGBoost · SHAP · matplotlib

## Key Results

*(Populated as analysis completes — placeholder section)*

- Panel regression coefficient tables → `results/`
- SHAP summary and dependence plots → `results/`

## Limitations

- Shareholder letters are not filed in a standardized SEC form; collection involves heterogeneous sources and some manual verification.
- Summary Compensation Table parsing from DEF 14A HTML is imperfect; values are spot-checked against proxy statements.
- Correlation ≠ causation: compensation is set by boards with information beyond letter tone.

## Author

**Vedant** — MS Applied Data Analytics, University of Detroit Mercy (Dec 2026)
Focus: compensation analytics, NLP, applied ML.

## License

MIT — see [LICENSE](LICENSE). Underlying SEC filings are public domain; shareholder letter texts remain property of their respective companies and are not redistributed here.
