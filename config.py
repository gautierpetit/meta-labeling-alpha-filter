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

# Ensure directories exist (optional, but convenient)
for path in [DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# === DATA SETTINGS ===
DATA_START_DATE = "2015-01-01"
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
    4,
    3,
)  # low:3,3;high:6,6, bestyet:5,5;4,3  negloss 1500%cumul, 0.9SR, logloss
MAX_HOLDING_PERIOD = 20


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
    "n_estimators": Integer(300, 2000),
    "learning_rate": Real(0.001, 0.2, prior="log-uniform"),
    "num_leaves": Integer(20, 200),
    "max_depth": Integer(3, 30),
    "min_child_samples": Integer(10, 100),
    "subsample": Real(0.6, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
    "reg_alpha": Real(1e-4, 10.0, prior="log-uniform"),
    "reg_lambda": Real(1e-4, 10.0, prior="log-uniform"),
    "scale_pos_weight": Real(0.5, 2.0),
    "min_split_gain": Real(1e-5, 1.0, prior="log-uniform"),
    "bagging_freq": Integer(0, 10),
}

NN_HP_SPACE = {
    "units1": [64, 128, 256, 512],
    "units2": [64, 128, 256],
    "units3": [64, 128],
    "units4": [32, 64],
    "l2_reg": [1e-6, 1e-5, 1e-4],
    "activation": ["relu", "selu", "tanh"],
    "n_hidden":{"min_value":1, "max_value":4, "step":1},
    "dropout": {"min_value": 0.0, "max_value": 0.3, "step": 0.05},
    "learning_rate": {"min_value": 1e-7, "max_value": 1e-4, "sampling": "log"},
}


LABEL_MAP = {-1: 0, 0: 1, 1: 2}

NN_TRAINING_PARAMS = {
    "epochs": 100,
    "batch_size": 128,
    "max_trials":50,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 1e-4,
}

# === META MODEL SETTINGS ===
TRAIN_END_DATE = "2020-12-31"
TEST_START_DATE = "2021-01-01"

CV_N_SPLITS = 3  # 5

RANDOM_SEARCH_ITER = 20  # 50
CV_SCORING = "neg_log_loss"  # "f1_weighted", "accuracy", "neg_log_loss"


META_PROBA_THRESHOLD = 0.5 # 0.45


# === BACKTESTING ===
BACKTEST_START_DATE = "2021-01-01"
TRANSACTION_COSTS = 0.001  # 10 bps

# === OUTPUT FILES ===
MISSING_DATA_REPORT = DATA_DIR / "missing_count.xlsx"
TICKER_AVAILABILITY_REPORT = DATA_DIR / "ticker_availability.xlsx"
PERFORMANCE_SUMMARY_XLSX = RESULTS_DIR / "performance_summary.xlsx"
BEST_MODEL_PATH = MODELS_DIR / "best_lgbm_model.pkl"
BEST_CAL_PATH = MODELS_DIR / "best_cal_model.pkl"
BEST_MLP = MODELS_DIR / "best_mlp.pkl"
MLP_CAL = MODELS_DIR / "mlp_calibrated.pkl"

CV_MODELS = MODELS_DIR / "cv_models.pkl"
