# Function to calculate cumulative features as described in the article


def calc_input_features(df, tickers, cols, time_step):
    """
    Calculate input features for all tickers, including:
    - intraday profit, NOT USED
    - daily profit,
    - turnover,
    - cumulative features.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.
        cols (list): Column names used to calculate intraday profit.
        time_step (int): Time step for cumulative calculations.

    Returns:
        dict of pd.DataFrame: Updated dictionary of DataFrames with all features.
    """

    df = calc_cumulative_features(df, tickers, time_step, cols)
    return df


def calc_cumulative_features(df, tickers, time_step, cols):
    """
    Calculate cumulative daily profit and turnover features for each ticker.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.
        time_step (int): Lookback window for cumulative calculations.

    Returns:
        dict of pd.DataFrame: Updated dictionary of DataFrames with cumulative features.
    """
    for ticker in tickers:
        df[ticker] = df[ticker].copy()
        df[ticker].loc[:, "Daily profit"] = (
            df[ticker]["Close"] - df[ticker]["Close"].shift(1)
        ) / df[ticker]["Close"].shift(1)
        df[ticker].loc[:, "Turnover"] = df[ticker]["Volume"] / df[ticker]["Close"]
        df[ticker].loc[:, "Cumulative profit"] = (
            df[ticker]["Daily profit"].rolling(window=time_step, min_periods=1).sum()
        )
        df[ticker].loc[:, "Cumulative turnover"] = (
            df[ticker]["Turnover"].rolling(window=time_step, min_periods=1).sum()
        )

        df[ticker].fillna({
            "Daily profit": 0,
            "Turnover": 0,
            "Cumulative profit": 0,
            "Cumulative turnover": 0
        }, inplace=True)
    return df
