# config.py
from scipy.stats import loguniform, randint, uniform
from pathlib import Path

# === GENERAL SETTINGS ===
RANDOM_STATE = 42
N_JOBS = -1  # Use all available cores

# === PATHS ===
ROOT_DIR = Path("")

DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Ensure directories exist (optional, but convenient)
for path in [DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# === DATA SETTINGS ===
DATA_START_DATE = "2015-01-01"
DATA_END_DATE = "2025-01-01"


SNP500_HISTORY_CSV = DATA_DIR / "S&P 500 Historical Components & Changes(03-10-2025).csv"
MASTER_PARQUET = DATA_DIR / "master.parquet"

FILTERED_PRICES = DATA_DIR / "S&P500_PIT.parquet"
FILTERED_VOLUMES = DATA_DIR / "S&P500_PIT_volumes.parquet"
FILTERED_HIGH = DATA_DIR / "S&P500_PIT_high.parquet"
FILTERED_LOW = DATA_DIR / "S&P500_PIT_low.parquet"
X = DATA_DIR / "X.parquet"
Y = DATA_DIR / "Y.parquet"

# === MODELING SETTINGS ===
TOP_QUANTILE = 0.9
TARGET_TP_THRESHOLD = 0.20
TARGET_SL_THRESHOLD = 0.05



# === HYPERPARAMETER SEARCH SPACE ===
HYPERPARAM_SPACE = {
    "n_estimators": randint(100, 1000),             # Number of boosting rounds
    "learning_rate": loguniform(0.005, 0.2),        # Smaller values favored
    "num_leaves": randint(10, 128),                 # Controls complexity
    "max_depth": randint(3, 12),                    # Tree depth
    "min_child_samples": randint(5, 100),           # Regularization
    "subsample": uniform(0.6, 0.4),                 # Row sampling
    "colsample_bytree": uniform(0.6, 0.4),          # Feature sampling
    "reg_alpha": loguniform(1e-4, 10),              # L1 regularization
    "reg_lambda": loguniform(1e-4, 10),             # L2 regularization
    "scale_pos_weight": uniform(0.8, 1.4),          # Class imbalance correction
}

# === META MODEL SETTINGS ===
TRAIN_END_DATE = "2020-12-31"
TEST_START_DATE = "2021-01-01"

CV_N_SPLITS = 5
RANDOM_SEARCH_ITER = 30
CV_SCORING = "roc_auc"

META_PROBA_THRESHOLD = 0.6

# === BACKTESTING ===
BACKTEST_START_DATE = "2021-01-01"

# === OUTPUT FILES ===
MISSING_DATA_REPORT = DATA_DIR / "missing_count.xlsx"
TICKER_AVAILABILITY_REPORT = DATA_DIR / "ticker_availability.xlsx"
PERFORMANCE_SUMMARY_XLSX = RESULTS_DIR / "performance_summary.xlsx"
BEST_MODEL_PATH = MODELS_DIR / "best_lgbm_model.pkl"
CV_MODELS = MODELS_DIR / "cv_models.pkl"
SHAP_VALUES_PARQUET = RESULTS_DIR / "shap_values.parquet"
