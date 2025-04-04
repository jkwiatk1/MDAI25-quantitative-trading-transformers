# Function to calculate cumulative features as described in the article
import logging
import pandas as pd


def calc_input_features(
    df,
    tickers,
    cols=["Daily profit", "Turnover"],
    time_step=20,
):
    """
    Calculate input features for all tickers, including:
    - intraday profit, NOT USED
    - daily profit,
    - turnover,
    - cumulative features.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.
        cols (list): Column names used for calculation.
        time_step (int): Time step for cumulative calculations.

    Returns:
        dict of pd.DataFrame: Updated dictionary of DataFrames with all features.
    """

    df = calc_cumulative_features(df, tickers, time_step, cols)
    return df


def calc_cumulative_features(df_dict, tickers, time_step, cols):
    """
    Calculate cumulative daily profit and turnover features for each ticker.

    Args:
        df_dict (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.
        time_step (int): Lookback window for cumulative calculations.
        cols (list): List of cumulative features to calculate.

    Returns:
        dict: A dictionary {ticker: pd.DataFrame} with calculated features.
    """
    if not cols:
        logging.warning(
            "No columns specified for cumulative features calculation. Returning original dictionary."
        )
        return df_dict

    output_df_dict = {}

    for ticker in tickers:
        if ticker not in df_dict:
            logging.warning(
                f"Ticker {ticker} not found in input dict for calc_cumulative_features. Skipping."
            )
            continue

        ticker_df = df_dict[ticker].copy()

        if "Daily profit" in cols:
            if "Close" in ticker_df.columns:
                daily_profit = (
                    ticker_df["Close"] - ticker_df["Close"].shift(1)
                ) / ticker_df["Close"].shift(1)
                ticker_df.loc[:, "Daily profit"] = daily_profit.fillna(0)
            else:
                logging.warning(
                    f"Column 'Close' not found for ticker {ticker}. Cannot calculate 'Daily profit'."
                )

        if "Turnover" in cols:
            if "Volume" in ticker_df.columns and "Close" in ticker_df.columns:
                valid_close = ticker_df["Close"].replace(0, pd.NA)
                turnover = ticker_df["Volume"] / valid_close
                ticker_df.loc[:, "Turnover"] = turnover.fillna(0)
            else:
                logging.warning(
                    f"Columns 'Volume' or 'Close' not found for ticker {ticker}. Cannot calculate 'Turnover'."
                )

        if "Cumulative profit" in cols:
            if "Daily profit" in ticker_df.columns:
                cumulative_profit = (
                    ticker_df["Daily profit"]
                    .rolling(window=time_step, min_periods=1)
                    .sum()
                    .fillna(0)
                )
                ticker_df.loc[:, "Cumulative profit"] = cumulative_profit
            else:
                ticker_df.loc[:, "Cumulative profit"] = 0
                logging.warning(
                    f"'Daily profit' not available for ticker {ticker}. Setting 'Cumulative profit' to 0."
                )

        if "Cumulative turnover" in cols:
            if "Turnover" in ticker_df.columns:
                cumulative_turnover = (
                    ticker_df["Turnover"]
                    .rolling(window=time_step, min_periods=1)
                    .sum()
                    .fillna(0)
                )
                ticker_df.loc[:, "Cumulative turnover"] = cumulative_turnover
            else:
                ticker_df.loc[:, "Cumulative turnover"] = 0
                logging.warning(
                    f"'Turnover' not available for ticker {ticker}. Setting 'Cumulative turnover' to 0."
                )

        # ticker_df = ticker_df.fillna(0)

        output_df_dict[ticker] = ticker_df

    return output_df_dict
