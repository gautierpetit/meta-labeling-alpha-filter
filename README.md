# Meta-Labeling Alpha Filter

**Enhancing Momentum Strategies via Stacked Ensembles, Meta-Labeling, and Probabilistic Execution Control**

A full end-to-end systematic trading framework that filters and sizes momentum signals using stacked LightGBM and deep neural networks to maximize risk-adjusted returns. Designed with hedge fund-grade rigor: point-in-time data integrity, nested cross-validation, Bayesian hyperparameter optimization, execution cost modeling, and explainability via SHAP.
---

## Project Motivation

Classic momentum strategies often degrade under real-world conditions due to signal noise, crowding, and execution slippage.
This framework attacks those weaknesses by:
   - Estimating conditional success probabilities for each raw momentum signal (meta-labeling).

   - Stacking calibrated models to capture both tree-based and neural-net representations.

   - Filtering and weighting trades using confidence thresholds, probability gaps, and volatility targeting.

   - Incorporating realistic execution costs directly into backtests.

The result is a more selective, risk-aware, and capital-efficient momentum portfolio.


---

## Pipeline Overview

The project is built around a robust, end-to-end pipeline:

1. **Point-in-Time Data Construction:** 
   - Historical S&P 500 constituents and prices, with survivorship/look-ahead bias eliminated.

2. **Feature Engineering:** 
   - Time-series: momentum (multi-horizon), volatility, skew, TA indicators (RSI, MACD, ATR, OBV, VWAP, etc.).
   - Cross-sectional: percentile ranks, z-scores, correlation, beta.
   - Macro: yield curve slope, 10y rates, VIX levels and stress regimes.
   - Market microstructure: liquidity proxies, volume surges, Amihud illiquidity.

3. **Labeling – Triple Barrier Method:**
   - Volatility-scaled take-profit, stop-loss, and max holding period.
   - Label grid search & distribution visualization to optimize signal quality.

4. **Primary Models:**
   - LightGBM: fast, interpretable tree-based learner.
   - MLP v1: tuned via Bayesian search over architecture, dropout, L2, LR.

5. **Meta-Features & Stacking:**
   - Combine model outputs (probabilities, predictions, probability gaps, agreement metrics) with select macro features.
   - MLP v2: trained on meta-features to filter base signals.

6. **Execution Layer:**
   - Probability threshold & confidence gap filtering.
   - Probability-weighted position sizing.
   - Optional inverted-logic mode (anti-crowding).
   - Volatility targeting & leverage caps.

7. **Backtesting & Risk Analytics:**
   - Turnover, transaction costs (long/short asymmetric), leverage, drawdown duration.
   - Benchmarks: SPY, standard momentum (long-only and long/short).
   - Explainability via SHAP (tree & deep models).

---

## Selected Results

Exact figures omitted here – in the repo’s results/ you’ll find full tables and plots.

**Sharpe improvement:** +X% vs. baseline momentum.

**Max drawdown reduction:** -25% relative to standard momentum.

**Hit rate boost:** +Y% for both long and short legs.

**Profit factor:** > Z.

**Trade count reduction:** −N%, focusing capital on high-conviction signals.




---

## 📉 SHAP Explainability

Comprehensive SHAP analyses ensure transparency and interpretability of models, enhancing trust and auditability.

![SHAP Summary Plot](figures/shap_summary_MLPV1.png)

---

## Tech Stack
**Python 3.11**

**Modeling:** LightGBM, TensorFlow/Keras, Keras Tuner, scikit-learn

**Explainability:** SHAP, Matplotlib, seaborn

**Data:** pandas, pyarrow, joblib

**Backtesting & Risk:** Custom evaluation module with hedge fund-style metrics



---

## Repo Structure

```
meta-labeling-alpha-filter/
├── config.py              # Config & hyperparameter spaces
├── features.py            # Feature generation
├── labeling.py            # Triple barrier labeling + label scans
├── modeling.py            # LightGBM training & calibration
├── mlp_modeling.py        # MLP tuning, training, stacking
├── strategy.py            # Momentum signal generation
├── sizing.py              # Prob-weighted sizing & vol targeting
├── evaluation.py          # Risk/performance analytics
├── main.py                # Full pipeline orchestration
...

```

---

## 📘 How to Run

 ```
git clone https://github.com/gautierpetit/meta-labeling-alpha-filter.git
cd meta-labeling-alpha-filter
pip install -r requirements.txt
python main.py

```

   utputs (in /results and /figures):

- Strategy & benchmark performance summaries
- Cumulative returns, drawdown, turnover, leverage plots
- SHAP explainability visuals


---

## About the Author

**Gautier Petit, MSc Finance (HEC Lausanne)** – Quant researcher specializing in systematic trading, meta-labeling, and ML-driven execution.
Actively seeking hedge fund opportunities in research, strategy, and systematic portfolio management.
 [LinkedIn](https://linkedin.com/in/gautierpetitch) | [GitHub](https://github.com/gautierpetit) 

---



## License

This project is available under the MIT License. See the [LICENSE](LICENSE) file.