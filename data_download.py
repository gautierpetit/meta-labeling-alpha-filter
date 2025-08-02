import logging
from typing import List, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_constituents(path: str, start_date: str) -> pd.Series:
    """
    Load and clean the constituents data from a CSV file.

    Args:
        path (str): Path to the CSV file containing constituents data.
        start_date (str): Start date for filtering the data.

    Returns:
        pd.Series: Cleaned and filtered constituents data.
    """
    df = pd.read_csv(path)
    constituents = pd.Series(
        [row.split(",") for row in df.iloc[:, 1]],
        index=pd.to_datetime(df.iloc[:, 0], format="%Y-%m-%d"),
    )
    last_date_before_start = constituents.index[constituents.index < start_date].max()
    constituents = constituents[constituents.index >= last_date_before_start]
    constituents.loc[config.DATA_END_DATE] = constituents.iloc[
        -1
    ]  # Carry forward for masking

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
    fix_map.update(
        {
            "YHOO": "AABA",  # YHOO became AABA after Verizon deal
            "CELG": "BMY",  # Acquired by BMY
            "MON": "BAYRY",  # Monsanto → Bayer
            "DWDP": "DOW",  # DowDuPont split
            "VIAB": "PARA",  # Viacom → Paramount
            "CBS": "PARA",  # CBS also → Paramount
            "ALTR": "INTC",  # Altera acquired by Intel
            "BRCM": "AVGO",  # Broadcom legacy
            "SNDK": "WDC",  # SanDisk acquired by Western Digital
            "LO": "BTI",  # Lorillard → Reynolds → BAT
            "TWC": "CMCSA",  # Time Warner Cable → Comcast
            "RAI": "BTI",  # Reynolds → BAT
            "TWTR": "X",  # Twitter now trades as X (if listed again)
            "COV": "MDT",  # Covidien → Medtronic
            "LLTC": "ADI",  # Linear → Analog Devices
            "ARG": "AIR",  # Airgas → Air Liquide (no ticker match but placeholder)
        }
    )

    return constituents.apply(lambda x: sorted(set(fix_map.get(t, t) for t in x)))


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
        df.columns = df.columns.levels[0]
        df.sort_index(axis=1, inplace=True)

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
