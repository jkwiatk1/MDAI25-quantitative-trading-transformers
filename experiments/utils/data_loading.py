import pandas as pd


# Function to fill in missing days with default values
def fill_missing_days(df, tickers, start_date, end_date):
    """
    Fill missing dates.

    - Fills missing weekends or other gaps in trading days.
    - Uses forward-fill for 'Close'.
    - Sets default values for missing fields as specified:
      - 'Adj Close' = 'Close'
      - 'High' = 'Close'
      - 'Low' = 'Close'
      - 'Open' = 'Close'
      - 'Volume' = 0
    """
    # Ensure the 'Date' column is a datetime object and set it as the index
    for ticker in tickers:
        df[ticker]["Date"] = pd.to_datetime(df[ticker]["Date"])
        df[ticker].set_index("Date", inplace=True)

        # Create a full range of dates from the minimum to the maximum in the dataset
        # end_date = df[ticker].index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Reindex the DataFrame to include all dates in the range
        df[ticker] = df[ticker].reindex(full_date_range)

        # Fill missing 'Close' values with the value from the previous day (forward-fill)
        df[ticker]["Close"].fillna(method="ffill", inplace=True)

        # Assign default values for other columns based on 'Close' or specific rules
        df[ticker]["Adj Close"].fillna(method="ffill", inplace=True)
        df[ticker]["High"].fillna(method="ffill", inplace=True)
        df[ticker]["Low"].fillna(method="ffill", inplace=True)
        df[ticker]["Open"].fillna(method="ffill", inplace=True)
        df[ticker]["Volume"].fillna(method="ffill", inplace=True)
        df[ticker]["ticker"].fillna(method="ffill", inplace=True)
    return df


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
        df (pd.DataFrame): A data frame containing financial data.
        tickers (list): List of tickers to be selected.
        cols (list): List of column names to be included.

    Returns:
        pd.DataFrame: The filtered data frame.
    """
    return {ticker: data[cols] for ticker, data in df.items() if ticker in tickers}
