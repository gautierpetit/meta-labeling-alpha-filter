import pandas as pd
import yfinance as yf
from tqdm import tqdm
import config

# === Load S&P 500 historical components ===
constituents_df = pd.read_csv(
    config.DATA_DIR / "S&P 500 Historical Components & Changes(03-10-2025).csv"
)

# Parse constituent list for each date and convert index to datetime
constituents = pd.Series(
    [row.split(",") for row in constituents_df.iloc[:, 1]],
    index=pd.to_datetime(constituents_df.iloc[:, 0], format="%Y-%m-%d")
)

# Drop data before 2015-01-01
#FIXME: get constituents consistent with config file
constituents = constituents[constituents.index >= "2014-12-24"]
constituents["2025-01-01"] = constituents["2024-12-23"]  # carry forward for masking

# === Extract unique tickers ===
tickers = sorted(set(pd.Series(constituents.explode())))

# Fix tickers that changed names or were delisted
fix_map = {
    "BF.B": "BF-B", "BRK.B": "BRK-B", "AABA": "YHOO", "WFM": "AMZN", "WCG": "CVS",
    "UA": "UAA", "RTN": "RTX", "APC": "OXY", "DNB": "DNB", "JOY": "CAT",
    "LVLT": "LUMN", "HSP": "ABBV", "CBS": "PARA", "LLL": "LHX", "FISV": "FI",
    "HCBK": "MTB", "COV": "MDT", "FB": "META"
}

# Apply ticker fixes
constituents = constituents.apply(lambda x: sorted(set(fix_map.get(t, t) for t in x)))
tickers = sorted(set(fix_map.get(t, t) for t in tickers))

# === Download historical OHLCV data from Yahoo Finance ===
master_df = yf.download(
    tickers=tickers,
    start=config.DATA_START_DATE,
    end=config.DATA_END_DATE,
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    actions=False,
    progress=False,
    threads=True,
)

# Save full raw data
master_df.to_parquet(config.MASTER_PARQUET, engine="pyarrow")

# === Extract price and volume data ===
prices = master_df.loc[:, (slice(None), "Close")]
volumes = master_df.loc[:, (slice(None), "Volume")]
prices_high = master_df.loc[:, (slice(None), "High")]
prices_low = master_df.loc[:, (slice(None), "Low")]

# Flatten column multi-index
prices.columns = prices.columns.levels[0]
volumes.columns = volumes.columns.levels[0]
prices_high.columns = prices_high.columns.levels[0]
prices_low.columns = prices_low.columns.levels[0]

# Sort columns alphabetically
for df in [prices, volumes, prices_high, prices_low]:
    df.sort_index(axis=1, inplace=True)

# === Save missing data report ===
missing_count = prices.isna().sum().sort_values(ascending=False)
missing_count.to_excel(config.MISSING_DATA_REPORT, sheet_name="Missing Count")

# === Point-in-time masking ===
mask = pd.DataFrame(index=prices.index, columns=prices.columns, data=False)

for i in tqdm(range(len(constituents) - 1)):
    tickers = constituents.iloc[i]
    start = constituents.index[i]
    end = constituents.index[i + 1]
    mask.loc[start:end, tickers] = True

# Validate index/shape
assert prices.shape == mask.shape
assert (prices.index == mask.index).all()
assert (prices.columns == mask.columns).all()

# Apply mask to create point-in-time filtered prices/volumes
filtered_prices = prices.mask(~mask)
filtered_volumes = volumes.mask(~mask)
filtered_prices_high = prices_high.mask(~mask)
filtered_prices_low = prices_low.mask(~mask)

# Save filtered datasets
filtered_prices.to_parquet(config.FILTERED_PRICES, engine="pyarrow")
filtered_volumes.to_parquet(config.FILTERED_VOLUMES, engine="pyarrow")
filtered_prices_high.to_parquet(config.FILTERED_PRICES_HIGH, engine="pyarrow")
filtered_prices_low.to_parquet(config.FILTERED_PRICES_LOW, engine="pyarrow")

# === Quantify survivorship bias ===
valid_mask = mask & prices.notna()
availability_ratio = valid_mask.sum() / mask.sum()
availability_ratio.sort_values().to_excel(config.TICKER_AVAILABILITY_REPORT.xlsx)

print(f"Available data: {availability_ratio.mean():.2%} of the time")