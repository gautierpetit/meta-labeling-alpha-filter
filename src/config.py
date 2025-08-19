from pathlib import Path

from scipy.stats import loguniform, randint, uniform
from skopt.space import Categorical, Integer, Real

# === GENERAL SETTINGS ===
RANDOM_STATE = 42
N_JOBS = -1

# === PATHS ===
ROOT_DIR = Path("")

DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SHAP_VALUES_DIR = ROOT_DIR / "shap"
MLPV1_DIR = MODELS_DIR / "mlpv1"
RUNS_DIR = ROOT_DIR / "runs"
CLF_DIR = MODELS_DIR / "clf"

# Ensure directories exist
for path in [DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SHAP_VALUES_DIR, RUNS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# === DATA SETTINGS ===
DATA_START_DATE = "2010-01-01"
DATA_END_DATE = "2025-01-01"

SNP500_HISTORY_CSV = DATA_DIR / "S&P 500 Historical Components & Changes(03-10-2025).csv"
DGS10 = DATA_DIR / "DGS10.csv"
T10Y3M = DATA_DIR / "T10Y3M.csv"
MASTER_PARQUET = DATA_DIR / "master.parquet"

FILTERED_PRICES = DATA_DIR / "S&P500_PIT.parquet"
FILTERED_VOLUMES = DATA_DIR / "S&P500_PIT_volumes.parquet"
FILTERED_HIGH = DATA_DIR / "S&P500_PIT_high.parquet"
FILTERED_LOW = DATA_DIR / "S&P500_PIT_low.parquet"
SPY = DATA_DIR / "SPY.parquet"
VIX = DATA_DIR / "VIX.parquet"
X = DATA_DIR / "X.parquet"
Y = DATA_DIR / "Y.parquet"

# === MODELING SETTINGS ===
TOP_QUANTILE = 0.9
BOTTOM_QUANTILE = 0.1
PT_SL_FACTOR = (
    5,
    5,
)  # 5, 5
MAX_HOLDING_PERIOD = 63

# === META MODEL SETTINGS ===
FOLD1_START = "2010-01-01"  # '2011-02-28'
FOLD1_END = "2016-12-31"

FOLD2_START = "2017-01-01"
FOLD2_END = "2019-12-31"

FOLD3_START = "2020-01-01"
FOLD3_END = "2024-12-31"

CV_N_SPLITS = 3
CV_GAP = 63
RANDOM_SEARCH_ITER = 15


LABEL_MAP = {-1: 0, 0: 1, 1: 2}


# LightGBM — RandomizedSearchCV (fast, tight ranges)
HYPERPARAM_RANDOM = {
    "n_estimators": randint(200, 600),
    "learning_rate": loguniform(0.05, 0.2),
    "num_leaves": randint(20, 100),
    "max_depth": randint(3, 8),
    "min_child_samples": randint(50, 120),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "reg_alpha": loguniform(1e-4, 0.1),
    "reg_lambda": loguniform(1e-3, 1.0),
    "min_split_gain": loguniform(1e-6, 1e-3),
    "bagging_freq": randint(0, 3),
}
# LightGBM — BayesSearchCV (more sample-efficient)
# max_depth=-1, boosting_type="gbdt", objective="multiclass"

HYPERPARAM_BAYESIAN = {
    # capacity / learning dynamics
    "n_estimators": Integer(1100, 1500),  # best=1200
    "learning_rate": Real(0.006, 0.010, prior="log-uniform"),  # best≈0.008
    # tree shape
    "num_leaves": Integer(200, 240),  # best=228
    "min_child_samples": Integer(50, 70),  # best=59
    "min_split_gain": Real(3e-5, 2e-4, prior="log-uniform"),  # best≈7.9e-5
    # randomness / subsampling
    "subsample": Real(0.60, 0.80),  # best≈0.723
    "colsample_bytree": Real(0.58, 0.76),  # best=0.60
    "bagging_freq": Integer(1, 2),  # best=1
    "extra_trees": Categorical([True]),  # best=True (allow flip)
    # regularization
    "reg_alpha": Real(0.5, 2.0, prior="log-uniform"),  # best=1.0 (was bound)
    "reg_lambda": Real(2e-3, 2e-2, prior="log-uniform"),  # best≈4.8e-3
}

# MLP V1 — operates on raw features (broader, but still quick)
MLPV1_HP_SPACE = {
    "units1": [384, 512, 640, 768],
    "units2": [256, 384, 512],
    "units3": [96, 128, 192, 256],
    "n_hidden": {"min_value": 2, "max_value": 3, "step": 1},
    "dropout": {"min_value": 0.10, "max_value": 0.30, "step": 0.05},
    "l2_reg": [0.0, 1e-7, 1e-6, 1e-5],
    "activation": ["relu"],
    "learning_rate": {"min_value": 3e-4, "max_value": 3e-3, "sampling": "log"},
    "epochs": 100,
    "batch_size": 2048,
    "max_trials": 30,
    "batch_norm": True,
    "label_smoothing": 0.00,
}


NN_TRAINING_PARAMS = {
    "early_stopping_patience": 16,  # slightly tighter to save time
    "early_stopping_min_delta": 2e-4,
    "reduce_lr_patience": 4,
    "reduce_lr_factor": 0.5,
    "reduce_lr_min_lr": 1e-6,
}


# === BACKTESTING SETTINGS ===


LONG_ONLY = False  # Requires retraining


MIN_GAP = 0.1
TOP_K_PER_DAY = 3

PROB_WEIGHTING = True  # Use model probabilities to weight signals
WEIGHT_MODE = "margin"  # Options: "prob", "margin", "odds"
TARGET_VOL = 0.2  # -1 to turn off volatility targeting
VOL_SPAN = 63
LEVERAGE_CAP = 3  # -1 to turn off leverage cap

MIN_TRADE_EPS = 0.01
LAMBDA_BLEND = 0.2

LONG_SIDE_TC = 0.001  # 10 bps
SHORT_SIDE_TC = 0.002  # 20 bps
# Minimum gap between long and short probabilities to consider a signal valid

# === OUTPUT FILES ===
MISSING_DATA_REPORT = DATA_DIR / "missing_count.xlsx"
TICKER_AVAILABILITY_REPORT = DATA_DIR / "ticker_availability.xlsx"
PERFORMANCE_SUMMARY_XLSX = RESULTS_DIR / "performance_summary.xlsx"
CLF_PATH = CLF_DIR / "clf.pkl"
CLF_CAL_PATH = MODELS_DIR / "clf_cal.pkl"
MLPV1T = MODELS_DIR / "mlpv1t.keras"


# Side-aware thresholds (defaults keep old behavior)
META_PROBA_THRESHOLD_LONG = 0.45
META_PROBA_THRESHOLD_SHORT = 0.50  # slightly stricter for shorts

# Ranking mode for Top-K: "prob" (as-is), "edge", or "logit_edge"
META_SCORE_MODE = "edge"
