# Meta-Labeling Alpha Filter

Enhancing cross-sectional momentum with stacked meta-models, probabilistic execution, and institutional-grade controls.

[Python 3.11] [Deterministic & Reproducible] [No Look-Ahead]

A full end-to-end systematic trading framework that filters and sizes momentum signals using stacked LightGBM and deep neural networks to maximize risk-adjusted returns. Designed with hedge fund-grade rigor: point-in-time data integrity, nested cross-validation, Bayesian hyperparameter optimization, execution cost modeling, and explainability via SHAP.
---

## TL;DR (Out-of-Sample 2020-01-01 → 2024-12-31)
- Sharpe (gross / net): [fill] / [fill]
- Max Drawdown / Duration: [fill]% / [fill] days
- Hit Rate (Long / Short): [fill]% / [fill]%
- Executed trades: [fill] (post-filter)
- Turnover (avg daily): [fill]
- Costs: long = [fill] bps, short = [fill] bps; capacity ≤ [fill]% ADV per leg, weight cap = [fill]

What to notice: selective execution via meta-labeling improves risk-adjusted returns and cuts turnover vs. raw momentum; results hold across subperiods and parameter sweeps.

---

## Pipeline at a glance

1) Point-in-time data
   - Historical S&P 500 membership and PIT prices; macro (rates, VIX), SPY benchmark.

2) Features
   - Time-series: momentum (multi-horizon), realized vol, skew; TA (RSI, MACD, ATR, OBV, VWAP)
   - Cross-sectional: ranks, z-scores, beta, crowding proxies
   - Macro: yield curve slope, 10Y rates, VIX regimes

3) Labels (Triple-Barrier)
   - Volatility-scaled TP/SL, max holding period; grid scan and distribution checks

4) Models
   - LightGBM (random/Bayesian search) + calibrated probabilities
   - MLP v1 (Bayesian tuner): dropout, L2, LR, batch size; early stopping; StandardScaler

5) Meta-Features & Stacking
   - Combine model probs, preds, probability gaps, agreement metrics + select macro features
   - MLP v2 trained on meta-features for signal filtering

6) Execution & Sizing
   - Probability threshold + min_gap filter
   - Probability-weighted sizing; NORMAL / INVERTED logic
   - Vol targeting and leverage caps

7) Backtest & Risk
   - Strategy vs. benchmarks (SPY, vanilla momentum); turnover, costs, max DD, DD duration
   - Calibration curves, confusion matrices, learning curves

```
Data  →  Features  →  Labels  →  Base Models (LGBM, MLPv1)  →  Meta-Features  →  MLPv2 Filter
   \_________________________________________________________Stacking/Calibration________________/
                                         ↓
                                  Execution Layer
                               (threshold, gap, sizing,
                              vol targeting, leverage cap)
                                         ↓
                                 Backtest & Analytics
```

---

## Figures (generated) and selected results
- Label grid and balance
  - figures/heatmap_TP_SL_combined.png
  - figures/label_distribution_before_after.png
- Additional outputs
  - figures/calibration_curves/
  - figures/confusion_matrices/
  - figures/learning_curves/

> Tip: add your best cumulative returns and drawdown plots here once generated.

---

## Reproducibility & leakage controls
- PIT data and constituents; no survivorship/look-ahead bias
- Walk-forward folds and train-only scaling/HP tuning
- Global seeds; `TF_DETERMINISTIC_OPS=1`
- Artifacts written to versioned folders: `results/`, `figures/`, `models/`, `shap/`

---

## How to run (Windows / PowerShell)

1) Create env and install
```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Execute full pipeline
```
python main.py
```

Outputs:
- results/: performance summaries, classification reports
- figures/: plots (cumulative returns, drawdowns, turnover, SHAP)
- models/: trained and calibrated models

---

## Configuration knobs (edit `config.py`)
- Data window: `DATA_START_DATE`, `DATA_END_DATE`
- Execution logic: `LONG_ONLY`, `LOGIC in {"NORMAL","INVERTED"}`
- Meta-filtering: `META_PROBA_THRESHOLD`, `MIN_GAP`
- Sizing: `PROB_WEIGHTING`, `TARGET_VOL`, `LEVERAGE_CAP`, `VOL_SPAN`
- Labeling: `PT_SL_FACTOR`, `MAX_HOLDING_PERIOD`
- Search/Seeds/CPU: `RANDOM_STATE`, `N_JOBS`

> Defaults are conservative; sweep thresholds and vol targets for robustness.

---

## Robustness & sensitivity checks
- Threshold and `min_gap` sweeps; vol target and leverage caps
- NORMAL vs. INVERTED logic; long-only vs. long/short
- Subperiods: 2017–2019, 2020, 2022, 2023–2024

---

## Repo structure

```
meta-labeling-alpha-filter/
├── config.py              # Global config & hyperparameter spaces
├── data_loader.py         # PIT data & macro loaders
├── features.py            # Feature generation & meta-features
├── labeling.py            # Triple barrier labeling & scans
├── modeling.py            # LightGBM training + calibration
├── mlp_modeling.py        # MLP tuning/training (Keras, KerasTuner)
├── signals.py             # Meta-model filtering of raw signals
├── sizing.py              # Probability-weighted sizing, vol targeting
├── strategy.py            # Momentum signal construction (daily)
├── evaluation.py          # Backtests, drawdowns, turnover, costs
├── analysis.py            # Learning curves, SHAP, reports
├── main.py                # Orchestration (one-shot reproduce)
├── notebook/              # Exploration notebooks
├── figures/, models/, results/, shap/, data/
```

---

## Possible expansions

#TODO: 

---


## Tech stack
- Python 3.11
- Modeling: LightGBM, TensorFlow/Keras, Keras Tuner, scikit-learn
- Explainability: SHAP, matplotlib, seaborn
- Data: pandas, pyarrow, joblib

---

## Contact
Gautier Petit — Quant researcher (systematic trading, meta-labeling, ML-driven execution)
- LinkedIn: https://linkedin.com/in/gautierpetitch
- GitHub: https://github.com/gautierpetit

Open to hedge fund roles in research and systematic portfolio management.

---

## License
MIT — see LICENSE.