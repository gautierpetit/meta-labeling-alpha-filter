# config.py
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
    3,
    3,
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
RANDOM_SEARCH_ITER = 20  
CV_SCORING = "neg_log_loss" 


LABEL_MAP = {-1: 0, 0: 1, 1: 2}

# === HYPERPARAMETER SEARCH SPACE ===
HYPERPARAM_RANDOM = {
    "n_estimators": randint(800, 1500),
    "learning_rate": loguniform(0.001, 0.01),  
    "num_leaves": randint(40, 128),  
    "max_depth": randint(40, 60),  
    "min_child_samples": randint(40, 100),  
    "subsample": uniform(0.6, 1),  
    "colsample_bytree": uniform(0.6, 1.0), 
    "reg_alpha": loguniform(1e-2, 5),  
    "reg_lambda": loguniform(1e-2, 2),  
    "scale_pos_weight": uniform(1.0, 3.0),  
    "min_split_gain": loguniform(1e-6, 1e-2), 
    "bagging_freq": randint(1, 10),  
}

HYPERPARAM_BAYESIAN = {
    # Better capacity but bounded runtime
    "n_estimators": Integer(300, 700),
    "learning_rate": Real(0.008, 0.06, prior="log-uniform"),
    "num_leaves": Integer(31, 127),
    "max_depth": Integer(7, 12),
    "min_child_samples": Integer(20, 80),
    "subsample": Real(0.7, 1.0),
    "colsample_bytree": Real(0.6, 1.0),
    "reg_alpha": Real(1e-5, 1e-1, prior="log-uniform"),
    "reg_lambda": Real(1e-5, 1e-1, prior="log-uniform"),
    "scale_pos_weight": Real(0.9, 1.5),
    "min_split_gain": Real(1e-6, 1e-2, prior="log-uniform"),
    "bagging_freq": Integer(0, 2),
}

MLPV1_HP_SPACE = {
    # More expressive than before but faster than very-wide nets
    "units1": [512, 1024, 1536],
    "units2": [256, 512, 768],
    "units3": [128, 256, 384],
    "units4": [64, 128],
    "units5": [32, 64],
    "n_hidden": {"min_value": 3, "max_value": 5, "step": 1},
    "dropout": {"min_value": 0.0, "max_value": 0.15, "step": 0.05},
    "l2_reg": [0.0, 1e-6, 5e-6, 1e-5, 5e-5],
    "activation": ["relu"],
    "learning_rate": {"min_value": 1e-7, "max_value": 1e-4, "sampling": "log"},
    "epochs": 100,
    "class_weight": {0: 0.8963260312726995, 1: 1.361415000116136, 2: 0.8697129545134175},
    "batch_size": 4096,
    "max_trials": 12,
    "batch_norm": True,
}

MLPV2_HP_SPACE = {
    # Slightly larger head than before; still light
    "units1": [96, 128, 192, 256],
    "units2": [48, 64, 96, 128],
    "units3": [24, 32, 48, 64],
    "units4": [16, 24, 32],
    "n_hidden": {"min_value": 2, "max_value": 3, "step": 1},
    "dropout": {"min_value": 0.0, "max_value": 0.30, "step": 0.05},
    "l2_reg": [0.0, 1e-6, 5e-6, 1e-5],
    "activation": ["relu"],
    "learning_rate": {"min_value": 1e-7, "max_value": 1e-4, "sampling": "log"},
    "epochs": 60,
    "class_weight": {0: 0.9019218562186837, 1: 1.3774958745874588, 2: 0.8581470059110768},
    "batch_size": 4096,
    "max_trials": 10,
    "batch_norm": False,
}

NN_TRAINING_PARAMS = {
    "early_stopping_patience": 8,   # slightly tighter to save time
    "early_stopping_min_delta": 1e-4,
}


# === BACKTESTING SETTINGS ===

LONG_SIDE_TC = 0.001  # 10 bps
SHORT_SIDE_TC = 0.002  # 20 bps

LONG_ONLY = False  # Requires retraining
LOGIC = "NORMAL"  # "NORMAL" or "INVERTED"
PROB_WEIGHTING = True  # Use model probabilities to weight signals
TARGET_VOL = -1  # -1 to turn off volatility targeting
VOL_SPAN = 20
LEVERAGE_CAP = -1  # -1 to turn off leverage cap
META_PROBA_THRESHOLD = 0.5 # 0.45
MIN_GAP = 0.2  # Minimum gap between long and short probabilities to consider a signal valid

# === OUTPUT FILES ===
MISSING_DATA_REPORT = DATA_DIR / "missing_count.xlsx"
TICKER_AVAILABILITY_REPORT = DATA_DIR / "ticker_availability.xlsx"
PERFORMANCE_SUMMARY_XLSX = RESULTS_DIR / "performance_summary.xlsx"
CLF_PATH = MODELS_DIR / "clf.pkl"
CLF_CAL_PATH = MODELS_DIR / "clf_cal.pkl"
MLPV1T = MODELS_DIR / "mlpv1t.keras"
MLPV2T = MODELS_DIR / "mlpv2t.keras"

