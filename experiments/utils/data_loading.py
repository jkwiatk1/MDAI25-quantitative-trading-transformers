import logging

import pandas as pd


# Function to fill in missing days with default values
def fill_missing_days(input_df_dict, tickers, start_date, end_date, freq="D"):
    """
    Fills missing dates for each ticker's DataFrame using forward-fill.

    Args:
        input_df_dict (dict): Dictionary {ticker: pd.DataFrame} with original data.
                              DataFrames should have a 'Date' column.
        tickers (list): List of tickers to process.
        start_date (pd.Timestamp): Start date for the desired date range.
        end_date (pd.Timestamp): End date for the desired date range.

    Returns:
        dict: A ictionary {ticker: pd.DataFrame} with missing dates filled.
    """
    output_df_dict = {}
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    for ticker in tickers:
        if ticker not in input_df_dict:
            logging.warning(
                f"Ticker '{ticker}' not found in input dictionary during fill_missing_days. Skipping."
            )
            continue

        df_ticker = input_df_dict[ticker].copy()

        try:
            if "Date" not in df_ticker.columns:
                logging.error(
                    f"Column 'Date' not found for ticker '{ticker}'. Cannot process."
                )
                continue
            df_ticker["Date"] = pd.to_datetime(df_ticker["Date"])
            df_ticker = df_ticker.drop_duplicates(subset=["Date"], keep="last")
            df_ticker.set_index("Date", inplace=True)
        except Exception as e:
            logging.error(
                f"Error processing 'Date' column or setting index for ticker '{ticker}': {e}"
            )
            continue

        if not df_ticker.index.is_monotonic_increasing:
            logging.warning(
                f"Index for ticker '{ticker}' is not monotonic increasing. Sorting index."
            )
            df_ticker = df_ticker.sort_index()

        cols_to_keep = [
            col
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"]
            if col in df_ticker.columns
        ]
        try:
            df_reindexed = df_ticker.reindex(full_date_range)[cols_to_keep]
        except Exception as e:
            logging.error(f"Error reindexing ticker '{ticker}': {e}")
            continue

        price_cols = [
            col
            for col in ["Open", "High", "Low", "Close", "Adj Close"]
            if col in df_reindexed.columns
        ]

        volume_col = ["Volume"] if "Volume" in df_reindexed.columns else []
        ticker_col = ["ticker"] if "ticker" in df_reindexed.columns else []
        df_filled = df_reindexed.bfill()
        df_filled = df_filled.ffill()

        if volume_col:
            df_filled[volume_col] = df_filled[volume_col].fillna(0)

        if ticker_col:
            if df_filled[ticker_col[0]].isnull().any():
                df_filled[ticker_col[0]] = df_filled[ticker_col[0]].ffill().bfill()
                if df_filled[ticker_col[0]].isnull().any():
                    df_filled[ticker_col[0]] = ticker

        final_check_cols = price_cols + volume_col
        if df_filled[final_check_cols].isnull().any().any():
            logging.warning(
                f"NaN values still present in ticker '{ticker}' after ffill/bfill. Check data source or filling logic."
            )

        output_df_dict[ticker] = df_filled.reset_index().rename(
            columns={"index": "Date"}
        )

    if not output_df_dict:
        logging.warning("fill_missing_days resulted in an empty dictionary.")

    return output_df_dict


# Data load
def load_finance_data_xlsx(path, is_from_yahoo=True):
    excel_data = pd.ExcelFile(path)
    data_dict = {}
    for sheet in excel_data.sheet_names:
        if is_from_yahoo == False:
            df = excel_data.parse(sheet, skiprows=2)
        else:
            df = excel_data.parse(sheet)
        df.columns = [
            "Date",
            "Adj Close",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
            "ticker",
        ]
        data_dict[sheet] = df
    return data_dict, excel_data.sheet_names


def prepare_finance_data(df, tickers, cols):
    """
        Prepares financial data by selecting specific tickers and columns.

    Args:
        df: A dict containing financial data.
        tickers (list): List of tickers to be selected.
        cols (list): List of column names to be included.

    Returns:
        pd.DataFrame: The filtered data frame.
    """
    return {ticker: data[cols] for ticker, data in df.items() if ticker in tickers}
