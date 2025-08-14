import logging
from typing import List, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm

import src.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_constituents(path: str, start_date: str) -> pd.Series:
    """
    Load and normalize S&P 500 constituents from a CSV file.

    Args:
        path (str): Path to the CSV file containing constituents data.
        start_date (str): Start date for filtering constituents.

    Returns:
        pd.Series: Normalized constituents data indexed by date.
    """
    df = pd.read_csv(path)
    raw = pd.Series(
        [row.split(",") for row in df.iloc[:, 1]],
        index=pd.to_datetime(df.iloc[:, 0], format="%Y-%m-%d"),
    )
    last_date_before_start = raw.index[raw.index < start_date].max()
    raw = raw[raw.index >= last_date_before_start]

    # 1) Generic Yahoo normalization: dot -> hyphen
    def yf_norm(t: str) -> str:
        return t.strip().replace(".", "-")

    # 2) True renames only (same security line)
    SAFE_RENAMES = {
        "FB": "META",
        "FISV": "FI",
        # keep dot/hyphen handled by yf_norm, so no need for BRK.B/BF.B here
    }

    out = []
    for tickers in raw:
        mapped = []
        seen = set()
        for t in tickers:
            t0 = yf_norm(t)
            t1 = SAFE_RENAMES.get(t0, t0)
            # Prevent collisions created by renames (e.g., CBS/VIAB->PARA was bad)
            if t1 in seen:
                # keep the original symbol instead of forcing a collision
                t1 = t0
            seen.add(t1)
            mapped.append(t1)
        out.append(sorted(set(mapped)))

    constituents = pd.Series(out, index=raw.index)

    # Optional: carry forward last set for masking end-boundary
    constituents.loc[config.DATA_END_DATE] = constituents.iloc[-1]

    return constituents



def download_market_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download historical market data for a list of tickers.

    Args:
        tickers (List[str]): List of ticker symbols.
        start (str): Start date for the data.
        end (str): End date for the data.

    Returns:
        pd.DataFrame: Historical market data.
    """
    logger.info(
        f"Downloading market data for {len(tickers)} tickers from {start} to {end}."
    )
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


def build_point_in_time_mask(
    prices: pd.DataFrame, constituents: pd.Series
) -> pd.DataFrame:
    """
    Build a point-in-time mask for the given prices and constituents.

    Args:
        prices (pd.DataFrame): Historical price data.
        constituents (pd.Series): Constituents data.

    Returns:
        pd.DataFrame: Point-in-time mask.
    """
    logger.info("Building point-in-time mask.")
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for i in tqdm(range(len(constituents) - 1), desc="Building point-in-time mask"):
        tickers = constituents.iloc[i]
        start = constituents.index[i]
        end = constituents.index[i + 1]
        mask.loc[start:end, tickers] = True
    return mask


def extract_ohlcv(
    master_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract OHLCV (Open, High, Low, Close, Volume) data from the master DataFrame.

    Args:
        master_df (pd.DataFrame): Master DataFrame containing market data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            DataFrames for Close, Volume, High, and Low prices.
    """
    logger.info("Extracting OHLCV data.")
    close = master_df.loc[:, (slice(None), "Close")]
    volume = master_df.loc[:, (slice(None), "Volume")]
    high = master_df.loc[:, (slice(None), "High")]
    low = master_df.loc[:, (slice(None), "Low")]

    for df in [close, volume, high, low]:
        df.columns = df.columns.get_level_values(0)
        df = df.sort_index(axis=1)  

    return close, volume, high, low


def save_filtered_data(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    mask: pd.DataFrame,
) -> None:
    """
    Save filtered OHLCV data to disk.

    Args:
        prices (pd.DataFrame): Filtered price data.
        volumes (pd.DataFrame): Filtered volume data.
        high (pd.DataFrame): Filtered high price data.
        low (pd.DataFrame): Filtered low price data.
        mask (pd.DataFrame): Point-in-time mask.
    """
    logger.info("Saving filtered data.")
    filtered_prices = prices.mask(~mask)
    filtered_volumes = volumes.mask(~mask)
    filtered_high = high.mask(~mask)
    filtered_low = low.mask(~mask)

    filtered_prices.to_parquet(config.FILTERED_PRICES, engine="pyarrow")
    filtered_volumes.to_parquet(config.FILTERED_VOLUMES, engine="pyarrow")
    filtered_high.to_parquet(config.FILTERED_HIGH, engine="pyarrow")
    filtered_low.to_parquet(config.FILTERED_LOW, engine="pyarrow")

    logger.info("Filtered data saved successfully.")


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """
    De-duplicate columns in a DataFrame by averaging duplicate columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: De-duplicated DataFrame.
    """
    if df.columns.is_unique:
        return df
    logging.warning("De-duplicating duplicate tickers...")
    return df.T.groupby(level=0).mean().T


def main() -> None:
    """
    Main function to orchestrate the data processing pipeline.
    """
    logger.info("Starting data processing pipeline.")

    # === Load & clean constituents ===
    constituents = load_constituents(
        config.DATA_DIR / "S&P 500 Historical Components & Changes(03-10-2025).csv",
        start_date=config.DATA_START_DATE,
    )
    tickers = sorted(set(t for sublist in constituents for t in sublist))

    # === Download historical data ===
    master_df = download_market_data(
        tickers, config.DATA_START_DATE, config.DATA_END_DATE
    )

    prices, volumes, prices_high, prices_low = extract_ohlcv(master_df)

    # De-duplicate columns
    prices  = _dedupe(prices)
    volumes = _dedupe(volumes)
    prices_high = _dedupe(prices_high)
    prices_low = _dedupe(prices_low)

    # 1) Kill non-sensical prices (<= 0 or too tiny) so pct_change won't explode
    prices = prices.mask(prices <= 0)
    prices = prices.mask(prices < 1e-3)  # yfinance near-zero glitch guard

    # 2) Inspect & neutralize absurd daily returns
    r = prices.pct_change(fill_method=None)
    spike_mask = r.abs() > 5.0  # > +500% in a day is almost surely bad
    if spike_mask.any().any():
        offenders = spike_mask.sum().sort_values(ascending=False).head(10)
        logging.warning("Clipping extreme returns; top offenders: %s", offenders.index.tolist())
        # null out just the offending points
        prices = prices.mask(spike_mask) 


    # === Download macro indicators ===
    logger.info("Downloading macro indicators.")
    spy = yf.download(
        "SPY", start=config.DATA_START_DATE, end=config.DATA_END_DATE, progress=False
    )["Close"]
    vix = yf.download(
        "^VIX", start=config.DATA_START_DATE, end=config.DATA_END_DATE, progress=False
    )["Close"]

    spy.to_parquet(config.SPY, engine="pyarrow")
    vix.to_parquet(config.VIX, engine="pyarrow")

    # === Save missing data report ===
    logger.info("Saving missing data report.")
    missing_count = prices.isna().sum().sort_values(ascending=False)
    missing_count.to_excel(config.MISSING_DATA_REPORT, sheet_name="Missing Count")

    # === Build point-in-time mask ===
    mask = build_point_in_time_mask(prices, constituents)


    # === Validate and save ===
    assert prices.shape == mask.shape, "Prices and mask shapes do not match."
    assert (prices.index == mask.index).all(), "Prices and mask indices do not match."
    assert (prices.columns == mask.columns).all(), (
        "Prices and mask columns do not match."
    )

    save_filtered_data(prices, volumes, prices_high, prices_low, mask)

    # === Quantify survivorship bias ===
    logger.info("Quantifying survivorship bias.")
    valid_mask = mask & prices.notna()
    availability_ratio = valid_mask.sum() / mask.sum()
    availability_ratio.sort_values().to_excel(
        config.TICKER_AVAILABILITY_REPORT.with_suffix(".xlsx")
    )
    logger.info(f"Average availability ratio: {availability_ratio.mean():.2%}")

    logger.info("Data processing pipeline completed successfully.")


if __name__ == "__main__":
    main()
