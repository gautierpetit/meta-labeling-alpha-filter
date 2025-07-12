import pandas as pd
import yfinance as yf
from tqdm import tqdm

import config


def load_constituents(path: str, start_date: str) -> pd.Series:
    df = pd.read_csv(path)
    constituents = pd.Series(
        [row.split(",") for row in df.iloc[:, 1]],
        index=pd.to_datetime(df.iloc[:, 0], format="%Y-%m-%d"),
    )
    constituents = constituents[constituents.index >= start_date]
    constituents["2025-01-01"] = constituents["2024-12-23"]  # Carry forward for masking

    fix_map = {
        "BF.B": "BF-B",
        "BRK.B": "BRK-B",
        "AABA": "YHOO",
        "WFM": "AMZN",
        "WCG": "CVS",
        "UA": "UAA",
        "RTN": "RTX",
        "APC": "OXY",
        "DNB": "DNB",
        "JOY": "CAT",
        "LVLT": "LUMN",
        "HSP": "ABBV",
        "CBS": "PARA",
        "LLL": "LHX",
        "FISV": "FI",
        "HCBK": "MTB",
        "COV": "MDT",
        "FB": "META",
    }

    return constituents.apply(lambda x: sorted(set(fix_map.get(t, t) for t in x)))


def download_market_data(tickers, start, end):
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        actions=False,
        progress=False,
        threads=True,
    )


def build_point_in_time_mask(prices, constituents):
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for i in tqdm(range(len(constituents) - 1), desc="Building point-in-time mask"):
        tickers = constituents.iloc[i]
        start = constituents.index[i]
        end = constituents.index[i + 1]
        mask.loc[start:end, tickers] = True
    return mask


def extract_ohlcv(master_df):
    close = master_df.loc[:, (slice(None), "Close")]
    volume = master_df.loc[:, (slice(None), "Volume")]
    high = master_df.loc[:, (slice(None), "High")]
    low = master_df.loc[:, (slice(None), "Low")]

    for df in [close, volume, high, low]:
        df.columns = df.columns.levels[0]
        df.sort_index(axis=1, inplace=True)

    return close, volume, high, low


def save_filtered_data(prices, volumes, high, low, mask):
    filtered_prices = prices.mask(~mask)
    filtered_volumes = volumes.mask(~mask)
    filtered_high = high.mask(~mask)
    filtered_low = low.mask(~mask)

    filtered_prices.to_parquet(config.FILTERED_PRICES, engine="pyarrow")
    filtered_volumes.to_parquet(config.FILTERED_VOLUMES, engine="pyarrow")
    filtered_high.to_parquet(config.FILTERED_PRICES_HIGH, engine="pyarrow")
    filtered_low.to_parquet(config.FILTERED_PRICES_LOW, engine="pyarrow")

    print("Filtered data saved.")


def main():
    # === Load & clean constituents ===
    constituents = load_constituents(
        config.DATA_DIR / "S&P 500 Historical Components & Changes(03-10-2025).csv",
        start_date="2014-12-24",
    )
    tickers = sorted(set(t for sublist in constituents for t in sublist))

    # === Download historical data ===
    print("Downloading market data...")
    master_df = download_market_data(
        tickers, config.DATA_START_DATE, config.DATA_END_DATE
    )

    prices, volumes, prices_high, prices_low = extract_ohlcv(master_df)

    # === Download macro indicators ===
    spy = yf.download(
        "SPY", start=config.DATA_START_DATE, end=config.DATA_END_DATE, progress=False
    )["Close"]
    vix = yf.download(
        "^VIX", start=config.DATA_START_DATE, end=config.DATA_END_DATE, progress=False
    )["Close"]

    spy.to_parquet(config.SPY, engine="pyarrow")
    vix.to_parquet(config.VIX, engine="pyarrow")

    # === Save missing data report ===
    missing_count = prices.isna().sum().sort_values(ascending=False)
    missing_count.to_excel(config.MISSING_DATA_REPORT, sheet_name="Missing Count")

    # === Build point-in-time mask ===
    mask = build_point_in_time_mask(prices, constituents)

    # === Validate and save ===
    assert prices.shape == mask.shape
    assert (prices.index == mask.index).all()
    assert (prices.columns == mask.columns).all()

    save_filtered_data(prices, volumes, prices_high, prices_low, mask)

    # === Quantify survivorship bias ===
    valid_mask = mask & prices.notna()
    availability_ratio = valid_mask.sum() / mask.sum()
    availability_ratio.sort_values().to_excel(
        config.TICKER_AVAILABILITY_REPORT.with_suffix(".xlsx")
    )
    print(f"Average availability ratio: {availability_ratio.mean():.2%}")


if __name__ == "__main__":
    main()
