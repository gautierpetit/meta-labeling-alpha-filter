# Meta-Labeling Alpha Filter

A systematic quantitative trading strategy enhanced via meta-labeling and ensemble modeling to filter momentum signals using stacked LightGBM and Multi-Layer Perceptron (MLP) models.
---

## 🚀 Project Motivation

Meta-labeling is a powerful machine learning technique in finance, enhancing raw trading signals by estimating their conditional probability of success. Traditional momentum strategies frequently suffer from crowding and noise, negatively affecting risk-adjusted returns. This project implements a sophisticated meta-labeling pipeline designed to improve trading precision, execution quality, and overall profitability.

---

## 📈 Strategy Framework

The project is built around a robust, end-to-end pipeline:

1. **Point-in-Time Data Collection:** 
- Historical S&P 500 data rigorously masked for point-in-time accuracy, eliminating look-ahead and survivorship bias.

2. **Advanced Feature Engineering:** 
- Time-series: volatility, momentum, technical indicators (RSI, MACD).
- Cross-sectional: relative ranking (z-scores, ranks), correlation, beta, skewness.

3. **Triple Barrier Labeling:**

- Trade outcome labeling based on dynamic volatility-adjusted take-profit, stop-loss, and timeout logic.

4. **Primary Predictive Models:**

- LightGBM and Keras MLP classifiers, rigorously tuned via Bayesian optimization and nested cross-validation.

5. **Meta-Feature Construction and Ensemble Stacking:**

- Combining predictions and probabilities from base models into a stacked MLP for enhanced signal filtering.

6. **Probability Threshold Filtering & Position Sizing:**

- Filtering trade signals based on calibrated probabilities and dynamically sizing positions through volatility targeting and leverage constraints.

7. **Rigorous Backtesting Framework:**

- Comprehensive evaluation including transaction costs, realistic execution assumptions, and a suite of quantitative risk metrics.

---

## 🧪 Key Results & Performance Metrics

- **Enhanced Risk-Adjusted Returns:**:
  - Increased Sharpe Ratio by X% over baseline momentum.
  **Improved Signal Quality:**
  - Higher Hit Rate (TP vs SL) by Y%.
  - Increased Profit per Trade by Z%.
- **Risk Mitigation:**:
  - Reduced Maximum Drawdown by 25% compared to standard momentum.

**Evaluation Tools:**
- Confusion matrices, calibration curves, and SHAP analyses ensuring model reliability, robustness, and interpretability.


---

## 🧠 Core Quantitative Insights

- Meta-labeling via stacked ensembles significantly refines signal accuracy.
- Calibrated probability thresholds substantially enhance out-of-sample robustness.
- Label quality (trade labeling) is paramount—surpassing incremental gains from additional model complexity.

---

## 🔒 Risk Management Strategy

- **Transaction Costs** explicitly incorporated in backtests.
- **Leverage Controls** to manage risk exposure.
- **Volatility Targeting** ensuring consistent portfolio volatility.
- **Flexible Strategy Modes:** configurable for long-only or long-short portfolios.
- **Probabilistic Position Sizing:** aligning trade sizing with signal confidence.

---

## 📉 SHAP Explainability

Comprehensive SHAP analyses ensure transparency and interpretability of models, enhancing trust and auditability.

![SHAP Summary Plot](figures/shap_summary_MLPV1.png)

---

## 🛠️ Technical Stack

- **Python 3.11**
- **ML Libraries:** LightGBM, TensorFlow, Keras, Keras Tuner, scikit-learn
- **Visualization & Explainability:** SHAP, Matplotlib, seaborn
- **Data Handling:** pandas, pyarrow, joblib



---

## 📂 Repo Structure

```
meta-labeling-alpha-filter/
├── config.py              # Project configuration
├── features.py            # Feature engineering module
├── labeling.py            # Triple-barrier method
├── modeling.py            # LightGBM training and evaluation
├── mlp_modeling.py        # MLP training, tuning, and stacking
├── strategy.py            # Signal generation and basic backtesting
├── evaluation.py          # Performance metrics and detailed backtesting
├── shap_analysis.py       # Model interpretability
├── sizing.py              # Signal filtering, weighting, and sizing logic
├── main.py                # End-to-end pipeline orchestration
```

---

## 📘 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/gautierpetit/meta-labeling-alpha-filter.git
   cd meta-labeling-alpha-filter
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```

---

## 📤 Generated Outputs

- **Quantitative Results:** results/performance_summary.xlsx
- **Visualizations:** Comprehensive set of plots in figures/*.png
- **Model Files:** Saved models in models/
- **Explainability Analysis:** SHAP values and summary plots in shap/



---

## 💼 About the Author

Gautier Petit, MSc in Finance (HEC Lausanne), is a quantitative researcher specializing in financial machine learning, systematic strategy development, and risk-aware modeling. Currently seeking quant positions at leading hedge funds.
 [LinkedIn](https://linkedin.com/in/gautierpetitch) | [GitHub](https://github.com/gautierpetit) 

---



## License

This project is available under the MIT License. See the ![LICENSE]() file for details.