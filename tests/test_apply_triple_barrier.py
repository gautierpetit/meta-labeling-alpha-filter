import pandas as pd
from labeling import apply_triple_barrier


def test_apply_triple_barrier():
    # Simulate sample data
    dates = pd.date_range(start="2023-01-01", periods=10)

    prices = pd.DataFrame(
        {
            "AAPL": [100, 102, 104, 103, 101, 105, 107, 106, 108, 110],
            "MSFT": [200, 198, 202, 204, 203, 205, 207, 206, 208, 210],
        },
        index=dates,
    )

    daily_signals = pd.DataFrame(
        {
            "AAPL": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "MSFT": [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        index=dates,
    )

    volatility = pd.DataFrame(
        {
            "AAPL": [0.02] * 10,
            "MSFT": [0.01] * 10,
        },
        index=dates,
    )

    pt_sl_factor = (2, 2)  # Take-profit and stop-loss multipliers
    max_holding_period = 5

    # Apply the triple barrier method
    labels, label_times = apply_triple_barrier(
        prices=prices,
        daily_signals=daily_signals,
        volatility=volatility,
        pt_sl_factor=pt_sl_factor,
        max_holding_period=max_holding_period,
    )

    # Assertions to validate behavior
    assert labels.loc["2023-01-01", "AAPL"] == 1, "AAPL should hit take-profit."
    assert labels.loc["2023-01-01", "MSFT"] == -1, "MSFT should hit stop-loss."
    assert pd.notna(label_times.loc[("2023-01-01", "AAPL"), "t1"]), (
        "AAPL exit date should be recorded."
    )
    assert pd.notna(label_times.loc[("2023-01-01", "MSFT"), "t1"]), (
        "MSFT exit date should be recorded."
    )

    print("All tests passed!")


if __name__ == "__main__":
    test_apply_triple_barrier()
