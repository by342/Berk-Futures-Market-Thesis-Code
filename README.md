# Cryptocurrency Return Predictability and Futures-Market Pressure

Code for my MSc thesis at Rotterdam School of Management (2026), supervised by Dr. Shiwei Ye.
It replicates Liu and Tsyvinski, "Risks and Returns of Cryptocurrency" (*Review of Financial
Studies*, 2021), and extends it to ask whether positioning in regulated crypto futures
markets predicts returns. The full thesis is in [`Paper/`](Paper/).

## Abstract

I replicate and extend the cryptocurrency-return predictability framework of Liu and
Tsyvinski (2021). Despite differences in sample composition and the unavailability of
wallet-user data, the replicated coin market index closely reproduces the original study's
main descriptive findings, where cryptocurrency returns exhibit substantial volatility, fat
tails, positive skew, and extreme price movements. I also find substantial support for
time-series momentum, and Google attention. The network-adoption results remain sensitive to
data construction, where individual predictors are noisy and the principal component of
network factors positively predicts subsequent returns. In the extension, I investigate
whether regulated cryptocurrency futures markets contain incremental information for
forecasting short-horizon returns beyond current returns and Google search-based investor
attention. Using Bitcoin and Ethereum futures data obtained from the Commodity Futures
Trading Commission (CFTC), I construct a Cryptocurrency Futures Pressure Index (CFPI) from
the momentum-adjusted open-interest growth, futures basis, and gross asset-manager
participation through static and dynamic Principal Component Analysis (PCA). The static CFPI
positively predicts subsequent cryptocurrency returns over the one to four week horizons. A
stricter dynamic specification, which estimates its inputs using lagged information, retains
predictive power primarily over the first two weeks. Among the individual components,
momentum-adjusted open-interest growth provides the clearest forecasting signal; whereas the
futures-basis is positively associated with future returns, and gross asset manager
participation adds little forecasting power. Overall, I find that cryptocurrency futures
markets contain short-horizon predictive information beyond established momentum and
attention channels, with the strength of the evidence depending on index construction
methodology.

## Repository

| File / folder | Contents |
|---|---|
| `01_build_dataset.ipynb` | Builds the daily panel for the replication. Includes blockchain activity, S&P 500 returns, Google Trends, and the value-weighted coin market index. |
| `02_replication.ipynb` | Replicates Liu–Tsyvinski (2021) tables and figures. |
| `03_cfpi_construction.ipynb` | Cleans the CFTC reports, builds the futures basis and positioning components, constructs the CFPI and runs the main predictive regressions. |
| `cfpi_variants.py` | Builds the alternative CFPI specifications, including momentum-adjusted open-interest growth and dynamic PCA, runs the predictive regressions and writes the LaTeX tables. |
| `Robustness/` | No-look-ahead specification and rolling-window sensitivity. Both scripts use `cfpi_variants.py`. |
| `Ex-Post/` | Includes the graphs I produced following my defense given the feedback I received. These were not included in the paper.|
| `Tables/`, `Figures/` | Outputs reported in the thesis.|

Run order:
- Replication: 01 → 02
- Extension: 01 → 03 → `cfpi_variants.py` → `Robustness/`

## Data

The raw data are not included because of their size and licensing. Sources:

- **Coin prices and market caps:** CoinMarketCap historical data (coin market index)
- **Blockchain activity:** Coin Metrics Community API (active addresses, transaction counts)
- **Stock market, spot and futures basis data:** Yahoo Finance (S&P 500)
- **Investor attention:** Google Trends search interest by coin
- **Futures positioning:** CFTC Traders in Financial Futures reports, 2017–2025

## Requirements

`pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`,
`yfinance`, `coinmetrics-api-client`, `xlrd` and `openpyxl`.

## Contact

Berk Yılmaz · by342@cornell.edu
