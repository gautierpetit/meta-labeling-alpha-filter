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
)  
MAX_HOLDING_PERIOD = 20

# === META MODEL SETTINGS ===
FOLD1_START = "2010-01-01"  # '2011-02-28'
FOLD1_END = "2016-12-31"

FOLD2_START = "2017-01-01"
FOLD2_END = "2019-12-31"

FOLD3_START = "2020-01-01"
FOLD3_END = "2024-12-31"

CV_N_SPLITS = 3  
RANDOM_SEARCH_ITER = 50  
CV_SCORING = "neg_log_loss" 


LABEL_MAP = {-1: 0, 0: 1, 1: 2}

# === HYPERPARAMETER SEARCH SPACE ===
HYPERPARAM_RANDOM = {
    "n_estimators": randint(600, 1200),
    "learning_rate": loguniform(0.003, 0.01),  
    "num_leaves": randint(40, 100),  
    "max_depth": randint(20, 40),  
    "min_child_samples": randint(20, 60),  
    "subsample": uniform(0.7, 1),  
    "colsample_bytree": uniform(0.7, 1.0), 
    "reg_alpha": loguniform(1.0, 10),  
    "reg_lambda": loguniform(0.01, 1),  
    "scale_pos_weight": uniform(1.0, 2.0),  
    "min_split_gain": loguniform(1e-5, 1e-2), 
    "bagging_freq": randint(1, 10),  
}

HYPERPARAM_BAYESIAN = {
    "n_estimators": Integer(600, 1200),
    "learning_rate": Real(0.002, 0.01, prior="log-uniform"),
    "num_leaves": Integer(40, 100),
    "max_depth": Integer(20, 40),  
    "min_child_samples": Integer(20, 60),
    "subsample": Real(0.7, 1.0),
    "colsample_bytree": Real(0.7, 1.0),
    "reg_alpha": Real(1.0, 10.0, prior="log-uniform"),
    "reg_lambda": Real(0.01, 1.0, prior="log-uniform"),
    "scale_pos_weight": Real(1.0, 2.0),
    "min_split_gain": Real(1e-5, 1e-2, prior="log-uniform"),
    "bagging_freq": Integer(1, 10),
}

NN_HP_SPACE = {
    "units1": [256, 512, 1024],
    "units2": [128, 256, 512],
    "units3": [64, 128, 256],
    "units4": [32, 64],
    "n_hidden": {"min_value": 3, "max_value": 4, "step": 1},
    "dropout": {"min_value": 0.05, "max_value": 0.35, "step": 0.05},
    "l2_reg": [1e-6, 1e-5, 1e-4],
    "activation": ["relu","swish"], 
    "learning_rate": {"min_value": 1e-7, "max_value": 1e-5, "sampling": "log"},
}

NN_TRAINING_PARAMS = {
    "epochs": 200, 
    "batch_size": 8192,
    "max_trials": 100,  
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 2e-4,
}


# === BACKTESTING SETTINGS ===

LONG_SIDE_TC = 0.001  # 10 bps
SHORT_SIDE_TC = 0.002  # 20 bps
LONG_ONLY = False
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
