# config.py
from pathlib import Path
from scipy.stats import loguniform, randint, uniform
from skopt.space import Categorical, Integer, Real

# === GENERAL SETTINGS ===
RANDOM_STATE = 42
N_JOBS = -1  # Use all available cores

# === PATHS ===
ROOT_DIR = Path("")

DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SHAP_VALUES_DIR = ROOT_DIR / "shap"

# Ensure directories exist
for path in [DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SHAP_VALUES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# === DATA SETTINGS ===
DATA_START_DATE = "2010-01-01"
DATA_END_DATE = "2025-01-01"

SNP500_HISTORY_CSV = (
    DATA_DIR / "S&P 500 Historical Components & Changes(03-10-2025).csv"
)
DGS10 = DATA_DIR / "DGS10.csv"
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
)  # low:3,3;high:6,6, bestyet:5,5;4,3  negloss 1500%cumul, 0.9SR, logloss
MAX_HOLDING_PERIOD = 20

# === META MODEL SETTINGS ===
FOLD1_START = "2010-01-01"  # '2011-02-28'
FOLD1_END = "2016-12-31"

FOLD2_START = "2017-01-01"
FOLD2_END = "2019-12-31"

FOLD3_START = "2020-01-01"
FOLD3_END = "2024-12-31"

CV_N_SPLITS = 3  # 5
RANDOM_SEARCH_ITER = 50  # 50
CV_SCORING = "neg_log_loss"  # "f1_weighted", "accuracy", "neg_log_loss"


LABEL_MAP = {-1: 0, 0: 1, 1: 2}

# === HYPERPARAMETER SEARCH SPACE ===
HYPERPARAM_RANDOM = {
    "n_estimators": randint(300, 2000),  # Wider boosting rounds range
    "learning_rate": loguniform(0.001, 0.2),  # Explore lower LR for slower convergence
    "num_leaves": randint(20, 200),  # Allow more complex trees
    "max_depth": randint(3, 30),  # Deeper trees if needed
    "min_child_samples": randint(10, 100),  # Tighter control on overfitting
    "subsample": uniform(0.6, 0.4),  # More sampling diversity
    "colsample_bytree": uniform(0.5, 0.5),  # Test lower values for feature selection
    "reg_alpha": loguniform(1e-4, 10),  # Broaden L1
    "reg_lambda": loguniform(1e-4, 10),  # Broaden L2
    "scale_pos_weight": uniform(0.5, 2.0),  # Very useful for class imbalance
    "min_split_gain": loguniform(1e-5, 1.0),  # Minimum gain to split
    "bagging_freq": randint(0, 10),  # Boosting randomization frequency
}

HYPERPARAM_BAYESIAN = {
    "n_estimators": Integer(400, 900),
    "learning_rate": Real(0.002, 0.01, prior="log-uniform"),
    "num_leaves": Integer(15, 40),
    "max_depth": Integer(20, 40),  # or fix to -1 (unlimited)
    "min_child_samples": Integer(40, 70),
    "subsample": Real(0.9, 1.0),
    "colsample_bytree": Real(0.8, 1.0),
    "reg_alpha": Real(5.0, 10.0),  # it wants heavy regularization
    "reg_lambda": Real(0.1, 1.0),
    "scale_pos_weight": Real(0.9, 1.2),
    "min_split_gain": Real(1e-5, 1e-2, prior="log-uniform"),
    "bagging_freq": Integer(5, 10),
}

NN_HP_SPACE = {
    "units1": [256, 512, 1024],
    "units2": [128, 256, 512],
    "units3": [64, 128, 256],
    "units4": [32, 64, 128],
    "n_hidden": {"min_value": 3, "max_value": 4, "step": 1},
    "dropout": {"min_value": 0.0, "max_value": 0.3, "step": 0.05},
    "l2_reg": [1e-5, 1e-4],
    "activation": ["relu", "gelu"],
    "learning_rate": {"min_value": 1e-6, "max_value": 5e-3, "sampling": "log"},
}

NN_TRAINING_PARAMS = {
    "epochs": 200,
    "batch_size": 8192,
    "max_trials": 100,  # 50
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 1e-4,
}


# === BACKTESTING SETTINGS ===

LONG_SIDE_TC = 0.001  # 10 bps
SHORT_SIDE_TC = 0.001  # 20 bps
LONG_ONLY = True
INVERT_SIGNALS = True
TARGET_VOL = 0.2
VOL_SPAN = 20
MAX_LEVERAGE = 4.0
META_PROBA_THRESHOLD = 0.45  # 0.45

# === OUTPUT FILES ===
MISSING_DATA_REPORT = DATA_DIR / "missing_count.xlsx"
TICKER_AVAILABILITY_REPORT = DATA_DIR / "ticker_availability.xlsx"
PERFORMANCE_SUMMARY_XLSX = RESULTS_DIR / "performance_summary.xlsx"
CLF_PATH = MODELS_DIR / "clf.pkl"
CLF_CAL_PATH = MODELS_DIR / "clf_cal.pkl"
MLPV1T = MODELS_DIR / "mlpv1t.pkl"
MLPV1 = MODELS_DIR / "mlpv1.pkl"
MLPV2T = MODELS_DIR / "mlpv2t.pkl"
MLPV2 = MODELS_DIR / "mlpv2.pkl"
CV_MODELS = MODELS_DIR / "cv_models.pkl"
