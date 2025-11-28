import logging
import pandas as pd


def fill_missing_days(input_df_dict, tickers, start_date, end_date, freq="D"):
    """
    Fill missing dates for each ticker using forward-fill.

    Args:
        input_df_dict: Dictionary {ticker: pd.DataFrame} with 'Date' column
        tickers: List of tickers to process
        start_date: Start date for date range
        end_date: End date for date range
        freq: Date frequency (default 'D' for daily)

    Returns:
        Dictionary {ticker: pd.DataFrame} with missing dates filled
    """
    output_df_dict = {}
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    for ticker in tickers:
        if ticker not in input_df_dict:
            logging.warning(f"Ticker '{ticker}' not found. Skipping.")
            continue

        df_ticker = input_df_dict[ticker].copy()

        try:
            if "Date" not in df_ticker.columns:
                logging.error(f"'Date' column not found for ticker '{ticker}'")
                continue
            df_ticker["Date"] = pd.to_datetime(df_ticker["Date"])
            df_ticker = df_ticker.drop_duplicates(subset=["Date"], keep="last")
            df_ticker.set_index("Date", inplace=True)
        except Exception as e:
            logging.error(f"Error processing ticker '{ticker}': {e}")
            continue

        if not df_ticker.index.is_monotonic_increasing:
            logging.warning(f"Sorting non-monotonic index for ticker '{ticker}'")
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
            logging.warning(f"NaN values remain in ticker '{ticker}' after fill")

        output_df_dict[ticker] = df_filled.reset_index().rename(columns={"index": "Date"})

    if not output_df_dict:
        logging.warning("No tickers processed in fill_missing_days")

    return output_df_dict



def load_finance_data_xlsx(path, is_from_yahoo=True):
    """
    Load financial data from Excel file.
    
    Args:
        path: Path to Excel file
        is_from_yahoo: If False, skip first 2 rows
    
    Returns:
        Tuple of (data_dict, sheet_names)
    """
    excel_data = pd.ExcelFile(path)
    data_dict = {}
    for sheet in excel_data.sheet_names:
        df = excel_data.parse(sheet) if is_from_yahoo else excel_data.parse(sheet, skiprows=2)
        df.columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume", "ticker"]
        data_dict[sheet] = df
    return data_dict, excel_data.sheet_names


def prepare_finance_data(df, tickers, cols):
    """
    Filter financial data by tickers and columns.

    Args:
        df: Dictionary of financial data
        tickers: List of tickers to select
        cols: List of column names to include

    Returns:
        Filtered dictionary {ticker: DataFrame}
    """
    return {ticker: data[cols] for ticker, data in df.items() if ticker in tickers}


def get_tickers(config):
    """
    Extract ticker list from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of ticker symbols
    """
    if config["data"].get("yahoo_data", False):
        try:
            ticker_file = config["data"]["tickers"]
            tickers_df = pd.read_csv(ticker_file)
            if "Ticker" not in tickers_df.columns:
                logging.error(f"Ticker file must contain 'Ticker' column: {ticker_file}")
                exit(1)
            return tickers_df["Ticker"].tolist()
        except FileNotFoundError:
            logging.error(f"Ticker file not found: {ticker_file}")
            exit(1)
        except Exception as e:
            logging.error(f"Error reading ticker file: {e}")
            exit(1)
    elif isinstance(config["data"]["tickers"], list):
        return config["data"]["tickers"]
    elif isinstance(config["data"]["tickers"], str):
        try:
            with open(config["data"]["tickers"], "r") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error(f"Ticker file not found: {config['data']['tickers']}")
            exit(1)
        except Exception as e:
            logging.error(f"Error reading ticker file: {e}")
            exit(1)
    else:
        logging.error("Invalid tickers format. Expected list or file path.")
        exit(1)
