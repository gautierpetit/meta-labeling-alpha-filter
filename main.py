import config
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from tqdm import tqdm
from utils import (backtest_strategy, filter_signals_with_meta_model,
                   generate_momentum_signals, get_trade_outcomes)

"""
Theme: “Turning noisy signals into reliable trades through predictive filtering”
Objective: Build a meta-model that learns when your base strategy is likely to succeed, effectively filtering out low-quality trades and boosting risk-adjusted performance.
"""

"""
1. Base Strategy ("Primary Model")

Data:
Get daily prices on US stocks, SP500 constituents

Split the data into training and test sets

Momentum: 

Strategy that selects stocks based on their returns over the previous 12 months
and then holds them for 3 months

Generate binary signals: 1 = take trade, 0 = no trade

"""


"""
Classic Jegadeesh & Titman (1993) momentum strategy:

Monthly signal generation: At the end of each month, compute each stocks 12-month past return (excluding the most recent month — often called a 1-month gap).

Ranking: Rank all available stocks by their 12-month return.

Top decile selection: Select top 10% (or top N) as the winners.

Holding period: Hold the selected stocks for 3 months, equally weighted, with overlapping portfolios (i.e., rebalance monthly, but hold each signal for 3 months).
"""


# Load point-in-time S&P500 prices
prices = pd.read_parquet(config.FILTERED_PRICES)

# Resample to month-end prices
monthly_prices = prices.resample("ME").last()

# Compute 12-month momentum with a 1-month gap
momentum = monthly_prices.pct_change(12) - monthly_prices.pct_change(1)


monthly_signals = generate_momentum_signals(momentum)


# Expand monthly signals to daily frequency with 3-month rolling window
# For each month, keep the signal alive for 3 months
daily_signals = pd.DataFrame(index=prices.index, columns=prices.columns, data=0)

for date in monthly_signals.index:
    start = date + pd.offsets.MonthEnd(1)  # Start next month (skip 1-month gap)
    end = start + pd.offsets.MonthEnd(2)  # Hold for 3 months
    tickers = monthly_signals.columns[monthly_signals.loc[date] == 1]
    daily_signals.loc[start:end, tickers] = 1

# Mask signals to only be valid when data is available (avoid applying signals during NaN periods)
daily_signals = daily_signals.where(~prices.isna(), other=0)

"""
daily_signals: a DataFrame of shape (daily_dates, tickers) where:
    1 = selected based on momentum
    0 = not selected
    NaNs are avoided — signal only active when prices exist
"""


# Calculate daily returns based on daily prices
daily_returns = prices.pct_change()
# Apply signals to returns to get strategy returns per stock per day
strategy_returns = daily_returns * daily_signals

# Aggregate portfolio returns
n_positions = daily_signals.sum(axis=1).replace(0, np.nan)  # avoid div by zero
mom_returns = strategy_returns.sum(axis=1) / n_positions

# Cumulative performance of the momentum strategy
mom_cumulative = (1 + mom_returns.fillna(0)).cumprod()
mom_cumulative.plot(title="Momentum Strategy Performance", figsize=(12, 6))

# Benchmark: SPY ETF

spy = yf.download("SPY", start=config.DATA_START_DATE, end=config.DATA_END_DATE, progress=False)
spy_returns = spy["Close"].pct_change().squeeze()  # Daily returns of SPY
spy_cumulative = (1 + spy_returns.fillna(0)).cumprod().squeeze()


"""
2. Meta-Labeling Problem Setup
Define the true outcome of each trade:
Did it reach a profit target (TP) or hit a stop-loss (SL)?
Y = 1 if the trade was successful (TP hit), 0 if it was unsuccessful (SL hit)


X = features at signal time
features = price, volatility (historical or VIX), serial correlation, earnings dates (or surprises), market cap, volume (illiquidity), beta, sentiment, FF factors ?, autocorrelation, etc.

Label: 1 = trade was good (TP hit), 0 = bad (SL hit)

"""

# Get trade outcomes based on TP/SL criteria
Y = get_trade_outcomes(prices, daily_signals)


# Build X features

# Only keep dates with active signals
signal_dates = daily_signals[daily_signals == 1].stack().index  # (date, ticker)

# Pre-compute features across entire price matrix
log_prices = np.log(prices)
volatility_20d = prices.pct_change().rolling(20).std()
volatility_zscore = (volatility_20d - volatility_20d.mean()) / volatility_20d.std()
momentum_12m_1m = prices.pct_change(252) - prices.pct_change(21)
momentum_6m = prices.pct_change(126)
momentum_12m = prices.pct_change(252)
momentum_change = momentum_6m - momentum_12m
vol_adj_momentum = momentum_12m / volatility_20d
returns_1d = prices.pct_change()
returns_5d = prices.pct_change(5)
returns_20d = prices.pct_change(20)
price_max_1y = prices.rolling(252).max()
price_min_1y = prices.rolling(252).min()
price_percentile_1y = (prices - price_min_1y) / (price_max_1y - price_min_1y)

vix = yf.download("^VIX", start=config.DATA_START_DATE, end=config.DATA_END_DATE, progress=False)["Close"]
vix = vix.reindex(prices.index).ffill()
# Broadcast VIX across all tickers
vix_feature = pd.DataFrame(
    np.tile(vix.values.reshape(-1, 1), (1, prices.shape[1])),
    index=prices.index,
    columns=prices.columns,
)

volume = pd.read_parquet(config.FILTERED_VOLUMES)

# Measures persistence of returns — useful for detecting mean-reversion or trend continuation.

serial_corr_5d = daily_returns.rolling(5).apply(lambda x: x.autocorr(lag=1), raw=False)

# Measures price impact per unit of volume.
amihud_illiquidity = daily_returns.abs() / volume
illiquidity_zscore = (
    amihud_illiquidity - amihud_illiquidity.mean()
) / amihud_illiquidity.std()

# Useful for regime awareness — e.g., when to avoid high-beta trades.
rolling_beta = pd.DataFrame(index=prices.index, columns=prices.columns)
for ticker in tqdm(prices.columns, desc="Computing Beta"):
    r = daily_returns[ticker]
    cov = r.rolling(60).cov(spy_returns)
    var = spy_returns.rolling(60).var()
    rolling_beta[ticker] = cov / var


rsi = prices.apply(lambda x: RSIIndicator(close=x, window=14).rsi())

# Bollinger Z-Score (20d):
zscore = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()

day_of_week_sin = pd.DataFrame(
    np.sin(2 * np.pi * prices.index.dayofweek / 7), index=prices.index
)
day_of_week_sin = pd.concat([day_of_week_sin] * len(prices.columns), axis=1)
day_of_week_sin.columns = prices.columns

month_of_year_sin = pd.DataFrame(
    np.sin(2 * np.pi * prices.index.month / 12), index=prices.index
)
month_of_year_sin = pd.concat([month_of_year_sin] * len(prices.columns), axis=1)
month_of_year_sin.columns = prices.columns


prices_low = pd.read_parquet(config.FILTERED_LOW)
prices_high = pd.read_parquet(config.FILTERED_HIGH)

adx = pd.DataFrame(index=prices.index, columns=prices.columns)
for col in prices.columns:
    high = prices_high[col]
    low = prices_low[col]
    close = prices[col]
    adx[col] = ADXIndicator(high=high, low=low, close=close, window=14).adx()


"""
Keep in mind: features are based on rolling windows, so they will have NaNs at the start, hence X will not have data for the first 20 days (for volatility) and first 252 days (for momentum) even after reindexing from Y. Because of tickers joining after the start date, there may also be NaNs for some tickers in the first few months.
"""


# Collect feature DataFrames
features = {
    # Price Level
    "log_prices": log_prices,
    "returns_5d": returns_5d,
    "returns_20d": returns_20d,
    # Momentum
    "price_percentile_1y": price_percentile_1y,
    "momentum_12m_1m": momentum_12m_1m,
    # Volatility
    "volatility_20d": volatility_20d,
    "vix": vix_feature,
    # Correlation
    "serial_corr_5d": serial_corr_5d,
    "beta_60d": rolling_beta,
    # Liquidity
    "volume": volume,
    "amihud_illiquidity": amihud_illiquidity.rolling(5).mean(),
    # Trend Strength and Structure
    "rsi_14d": rsi,
    "adx_14d": adx,
    "momentum_change": momentum_change,
    # Time-Based Signals
    "day_of_week_sin": day_of_week_sin,
    "month_of_year_sin": month_of_year_sin,
    # Reversal/Mean-Reversion Signals
    "bollinger_zscore": zscore,
    "returns_1d": returns_1d,
    "volatility_zscore": volatility_zscore,
    # Event driven
    # days_to_next_earnings
    # days_since_last_earnings
    # prev_earnings_surprise
    # Fundamental
    # market_cap
    # sector_dummy
    # Macro
    # spy_rolling_corr
    # SKEW index, MOVE index
    # US 10y treasury yield
    # Engineered
    "vol_adj_momentum": vol_adj_momentum,
    "illiquidity_zscore": illiquidity_zscore,
    # beta_momentum
    # momentum_persistence
}

# Build X from stacked rows
X_rows = []

for date, ticker in tqdm(signal_dates, desc="Building X"):
    row = {"date": date, "ticker": ticker}
    for fname, fmat in features.items():
        value = (
            fmat.at[date, ticker]
            if (
                date in fmat.index
                and ticker in fmat.columns
                and not pd.isna(fmat.at[date, ticker])
            )
            else np.nan
        )

        row[fname] = value
    X_rows.append(row)

X = pd.DataFrame(X_rows).set_index(["date", "ticker"]).dropna()


# Step 1: Stack Y to get MultiIndex (date, ticker)
Y = Y.stack()

# Step 2: Drop trades with no outcome (e.g. neither TP nor SL hit)
Y = Y.dropna().astype(int)

# Step 3: Filter X to keep only rows with a known outcome
X = X.reindex(Y.index)


# This ensures we only keep rows where all features are available
# issue: X may have NaNs if some tickers joined after the start date due to rolling features
X = X.dropna()
Y = Y.loc[X.index]

# Safety checks
assert X.isnull().sum().sum() == 0  # No missing features
assert X.shape[0] == Y.shape[0]
assert (X.index == Y.index).all()


"""
3. Meta-Model
Use a classifier to predict the probability of success for each trade
RandomForestClassifier, XGBClassifier, etc...
May use hyperparameter tuning to optimize the model with GridSearchCV or RandomizedSearchCV
May use model calibration with CalibratedClassifierCV

Calculate feature importance to understand which features are most predictive
with permutation_importance or feature_importances_

Train the model on the training set.
Get accuracy on the training set.

Predict on the test set
Explain the model's predictions using SHAP or LIME


"""

# Scale features before training
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X), index=X.index, columns=X.columns
)

#FIXME: Split data using sklearn
X_train = X_scaled.loc[:"2020-12-31"]
Y_train = Y.loc[:"2020-12-31"]

X_test = X_scaled.loc["2021-01-01":]
Y_test = Y.loc["2021-01-01":]



"""X_train, X_test, Y_train, Y_test = train_test_split(
    X_train_scaled, Y, test_size=0.2, random_state=config.RANDOM_STATE
)"""

# Set up time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Define model
lgbm = LGBMClassifier(
    device_type="gpu",
    gpu_platform_id=0,
    gpu_device_id=0,
    random_state=config.RANDOM_STATE,
    class_weight="balanced",
    n_jobs=config.N_JOBS,
)




search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=config.HYPERPARAM_SPACE,
    n_iter=30,
    cv=tscv,
    scoring="roc_auc",
    verbose=2,
    n_jobs=config.N_JOBS,
    random_state=config.RANDOM_STATE,
)


search.fit(
    X_train,
    Y_train,
)

# Best estimator and score
print("Best Parameters:", search.best_params_)
print("Best CV Score:", search.best_score_)


# Use the best estimator to evaluate on test set
best_model = search.best_estimator_
test_score = best_model.score(X_test, Y_test)
print("Test Accuracy:", test_score)


# Initialize and train the LightGBM model
clf = search.best_estimator_
clf.fit(X_train, Y_train)


"""
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(estimator=clf, method='sigmoid', cv=tscv)
calibrated_clf.fit(X_train, Y_train)

"""


# Evaluate with cross_val_score (accuracy)
scores = cross_val_score(clf, X_train, Y_train, cv=tscv, scoring="accuracy")
print(f"Mean accuracy across splits: {np.mean(scores):.4f}")
print("All CV scores:", scores)

# Get accuracy on the training set
train_accuracy = clf.score(X_train, Y_train)
print(f"Training accuracy: {train_accuracy:.2f}")

# Predict on the test set
test_accuracy = clf.score(X_test, Y_test)
print(f"Test accuracy: {test_accuracy:.2f}")


y_test_proba = clf.predict_proba(X_test)[:, 1]
print("Test ROC AUC:", roc_auc_score(Y_test, y_test_proba))


joblib.dump(clf, config.BEST_MODEL_PATH)
joblib.load(config.BEST_MODEL_PATH)

cv_results = pd.DataFrame(search.cv_results_)
top_models = cv_results.sort_values("mean_test_score", ascending=False).head(5)
print(top_models[["params", "mean_test_score", "rank_test_score"]])


# Explain the model's predictions using SHAP or LIME


# Create a SHAP explainer
explainer = shap.Explainer(clf)
explanation = explainer(X_test)

# Global feature importance
shap.plots.beeswarm(explanation, show=False)
plt.savefig(config.FIGURES_DIR / "shap_beeswarm.png")
shap.plots.bar(explanation, show=False)
plt.savefig(config.FIGURES_DIR / "shap_bar.png")

# Local explanations for the first instance in the test set
shap.plots.force(explanation[0], show=False)
plt.savefig(config.FIGURES_DIR / "shap_force.png")
shap.plots.waterfall(explanation[0], show=False)
plt.savefig(config.FIGURES_DIR / "shap_waterfall.png")

# Dependence plots for specific features
shap.plots.scatter(explanation[:, "serial_corr_5d"], color=explanation, show=False)
plt.savefig(config.FIGURES_DIR / "shap_scatter.png")

shap_values_class1 = explanation.values
shap_values_df = pd.DataFrame(
    explanation, columns=X_test.columns, index=X_test.index
)
shap_values_df.to_parquet(config.SHAP_VALUES_PARQUET)


"""
4. Filtered Signal Generation

Use the meta-model to filter the base strategy's signals
Only take trades where meta-model predicts high success probability, can add confidence bands

Use the meta-model's predicted probability as a position size scaler ?

Apply thresholding on predicted probabilities (e.g., only take trades with P(success) > 0.6), and create a new filtered_signals DataFrame with the same shape as daily_signals.

Then compute the performance again using filtered_signals instead of daily_signals.

"""

filtered_signals = filter_signals_with_meta_model(
    daily_signals=daily_signals, clf=clf, X_meta_test=X_test, threshold=config.META_PROBA_THRESHOLD
)


"""
5. Backtesting and evaluation

Use the existing performance pipeline and summary function to compare:
base_strategy_returns
filtered_strategy_returns
spy_returns

Also count:
Trade count
Win rate (e.g. from outcomes)
Sharpe
Drawdown
Turnover

"""

# Apply filtered signals to returns to get strategy returns per stock per day
filtered_strategy_returns = daily_returns * filtered_signals
# Aggregate portfolio returns
n_positions_filtered = filtered_signals.sum(axis=1).replace(0, np.nan)
filtered_mom_returns = filtered_strategy_returns.sum(axis=1) / n_positions_filtered


# Get trade count and win rate
# 1. Get the (date, ticker) index of trades taken by the filtered strategy
filtered_trade_idx = filtered_signals.stack()[filtered_signals.stack() == 1].index

# 2. Subset Y to those trades
filtered_outcomes = Y.loc[Y.index.isin(filtered_trade_idx)]

# 3. Compute trade count and win rate
trade_count = len(filtered_outcomes)
win_rate = filtered_outcomes.mean()  # Since 1 = TP, 0 = SL


summary_meta, summary_spy, summary_mom = backtest_strategy(
    strategy_returns=filtered_mom_returns,
    bench_spy=spy_returns,
    bench_mom=mom_returns,
    name="Meta-Filtered Momentum",
    trade_count=trade_count,
    win_rate=win_rate,
    plot=True,
)

summary_meta["Trade Count"] = trade_count
summary_meta["Win Rate"] = f"{win_rate:.2%}"

summary = pd.concat([summary_mom, summary_spy, summary_meta], axis=1)
summary.columns = ["Standard Momentum", "SPY", "Meta-Filtered Momentum"]

summary.to_excel(config.PERFORMANCE_SUMMARY_XLSX)


"""
Idea stack:



Test other alpha signals (reversal, value, quality) using the same framework
→ Adds robustness and variety to signal generation.

Refine outcomes to multi-class (+1 = TP, -1 = SL, 0 = hold)
→ Useful for richer modeling (e.g., ordinal classifiers, directional prediction).

Model calibration (e.g., CalibratedClassifierCV)
→ Important if you're going to threshold on predicted probability.

Feature selection techniques
→ Helps reduce noise, speed up training, and improve generalization.

Ensemble models
→ Stacking or blending classifiers can boost performance.

Bayesian optimization for hyperparameter tuning
→ More efficient than grid/random search; use optuna, skopt, etc.

Save pipeline stages to Parquet

DONE ----

Use LightGBM/XGBoost
→ Stronger predictive power; LightGBM is very effective in finance.
Faster training (especially on large datasets)
Native support for categorical variables (if added later)
Often better generalization than random forests
Can handle class imbalance with class_weight="balanced"

Time series cross-validation (Rolling Windows / Expanding Windows)
→ Mimics real-world prediction, gives better view of model stability.

Hyperparameter tuning  (e.g., GridSearchCV, RandomizedSearchCV)
→ Improves model performance, especially for tree-based models.


"""
