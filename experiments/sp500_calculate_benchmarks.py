import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import argparse
import logging
import os

try:
    from utils.metrics import calculate_benchmark_metrics
except ImportError:
    # Definicja funkcji calculate_benchmark_metrics (jak w poprzednich odpowiedziach)
    def calculate_benchmark_metrics(daily_returns, risk_free_rate=0.0, trading_days_per_year=252):
        # ... (pełna implementacja funkcji) ...
        if not isinstance(daily_returns, pd.Series): daily_returns = pd.Series(daily_returns)
        daily_returns = daily_returns.dropna()
        if len(daily_returns) == 0:
            logging.warning("Benchmark daily returns series is empty after dropping NaNs.")
            nan_metrics = {'Cumulative Return (%)': np.nan, 'Annualized Return (%)': np.nan, 'Annualized Volatility (%)': np.nan, 'Annualized Sharpe Ratio': np.nan, 'Maximum Drawdown (%)': np.nan}
            start_idx = daily_returns.index.min() - pd.Timedelta(days=1) if isinstance(daily_returns.index, pd.DatetimeIndex) else 0
            return nan_metrics, pd.Series([1.0], index=[start_idx] if isinstance(daily_returns.index, pd.DatetimeIndex) else None)
        value_curve = (1 + daily_returns).cumprod()
        if isinstance(daily_returns.index, pd.DatetimeIndex): start_date_curve = daily_returns.index.min() - pd.Timedelta(days=1); start_idx = start_date_curve
        else: start_idx = -1
        value_curve = pd.concat([pd.Series([1.0], index=[start_idx] if start_idx != -1 else None), value_curve])
        cr = value_curve.iloc[-1] - 1.0; num_days = len(daily_returns); num_years = num_days / trading_days_per_year
        ar = (value_curve.iloc[-1] ** (1.0 / num_years)) - 1.0 if num_years > 0 and value_curve.iloc[-1] > 0 else 0.0
        av = daily_returns.std() * np.sqrt(trading_days_per_year)
        sr = (ar - risk_free_rate) / (av + 1e-8) if av > 1e-8 else np.nan
        peaks = value_curve.cummax(); drawdowns = (value_curve - peaks) / (peaks + 1e-8)
        mdd = drawdowns.min() if not drawdowns.empty else 0.0
        metrics = {'Cumulative Return (%)': cr * 100, 'Annualized Return (%)': ar * 100, 'Annualized Volatility (%)': av * 100, 'Annualized Sharpe Ratio': sr, 'Maximum Drawdown (%)': mdd * 100}
        return metrics, value_curve

def setup_logging(log_file, level=logging.INFO):
    """Konfiguruje logowanie."""
    # ... (bez zmian) ...
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()], datefmt="%Y-%m-%d %H:%M:%S")
    logging.info("Benchmark calculation script logging setup complete.")


def load_config(config_path):
    """Wczytuje konfigurację."""
    # ... (bez zmian) ...
    try:
        with open(config_path, "r") as f: config = yaml.safe_load(f)
        logging.info(f"Successfully loaded benchmark configuration from {config_path}")
        return config
    except Exception as e: logging.error(f"Error loading config {config_path}: {e}", exc_info=True); raise

def load_benchmark_data(file_path, index_col_name='S&P500', date_col='Date'):
    """Wczytuje i przetwarza dane benchmarku S&P 500."""
    # ... (bez zmian - użyj poprawionej wersji) ...
    try:
        df = pd.read_excel(file_path, header=0, parse_dates=[date_col], decimal=',')
        if date_col not in df.columns: raise ValueError(f"Date column '{date_col}' not found.")
        if index_col_name not in df.columns: raise ValueError(f"Benchmark data column '{index_col_name}' not found.")
        df = df.set_index(date_col); df = df.sort_index()
        df_selected = df[[index_col_name]].copy()
        if not pd.api.types.is_numeric_dtype(df_selected[index_col_name]):
            try: df_selected[index_col_name] = df_selected[index_col_name].astype(float)
            except ValueError: df_selected[index_col_name] = df_selected[index_col_name].astype(str).str.replace(r'[^\d\.]', '', regex=True).astype(float)
        df_selected['DailyReturn'] = df_selected[index_col_name].pct_change().fillna(0)
        return df_selected
    except Exception as e: logging.error(f"Error loading/processing benchmark data: {e}", exc_info=True); return None

def main_benchmarks(config, config_path_str: str): # Dodano config_path_str jako argument
    """Główna funkcja obliczająca metryki dla benchmarków."""
    benchmark_data_config = config['benchmark_data']
    portfolio_config = config.get('portfolio', {})
    test_period_config = config['test_period']
    test_start_date = pd.to_datetime(test_period_config['start_date'])
    test_end_date = pd.to_datetime(test_period_config['end_date'])
    output_dir_bench = Path(config['benchmark_output_dir'])
    output_dir_bench.mkdir(parents=True, exist_ok=True)

    log_file = output_dir_bench / "benchmark_calculation.log"
    setup_logging(log_file)
    logging.info("--- Starting Benchmark Calculation Script (S&P 500 Buy & Hold Only) ---")
    logging.info(f"Config path: {config_path_str}") # Logowanie ścieżki do configu

    # --- Metryki dla S&P 500 Buy & Hold ---
    logging.info("Calculating S&P 500 Buy & Hold metrics...")
    df_benchmark_raw = load_benchmark_data(benchmark_data_config['path'],
                                           benchmark_data_config['column_name'])

    if df_benchmark_raw is None: logging.error("Failed to load benchmark data. Exiting."); return

    df_benchmark_test = df_benchmark_raw[(df_benchmark_raw.index >= test_start_date) & (df_benchmark_raw.index <= test_end_date)]
    if df_benchmark_test.empty: logging.error("No benchmark data for the specified test period."); return
    logging.info(f"Benchmark data covers {len(df_benchmark_test)} days in the test period.")

    sp500_metrics, sp500_curve = calculate_benchmark_metrics(df_benchmark_test['DailyReturn'],
                                                          risk_free_rate=portfolio_config.get('risk_free_rate', 0.0))
    logging.info(f"S&P 500 Buy & Hold Metrics: {sp500_metrics}")

    # --- Zapisz Wyniki S&P 500 ---
    benchmark_summary = { "S&P500 (Buy & Hold)": sp500_metrics }
    benchmark_summary_df = pd.DataFrame(benchmark_summary).T

    # Zapisz tabelę metryk CSV
    benchmark_metrics_path_csv = output_dir_bench / "sp500_bh_metrics_summary.csv"
    benchmark_summary_df.to_csv(benchmark_metrics_path_csv)
    logging.info(f"S&P 500 metrics summary saved to CSV: {benchmark_metrics_path_csv}")

    # --- Zapisz Wyniki do pliku TXT ---
    results_file_txt = output_dir_bench / "sp500_bh_metrics_summary.txt"
    try:
        with open(results_file_txt, "w") as f:
            f.write(f"Benchmark Evaluation Results\n")
            f.write(f"Config Path: {config_path_str}\n")
            f.write(f"Benchmark Data Path: {benchmark_data_config['path']}\n")
            f.write(f"Test Period: {test_start_date.date()} to {test_end_date.date()}\n")
            f.write(f"Number of Test Days: {len(df_benchmark_test)}\n")
            f.write("-" * 30 + "\nS&P 500 Buy & Hold METRICS:\n")
            if sp500_metrics:
                for key, value in sp500_metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
            else:
                f.write("Metrics calculation failed.\n")
        logging.info(f"S&P 500 evaluation results saved to TXT: {results_file_txt}")
    except Exception as e:
        logging.error(f"Error saving S&P 500 results to TXT: {e}")

    # --- Zapisz krzywą wartości ---
    # ... (kod zapisu krzywej do CSV - bez zmian) ...
    sp500_curve_path = output_dir_bench / "sp500_bh_value_curve.csv"
    if isinstance(sp500_curve.index, pd.DatetimeIndex): sp500_curve.to_csv(sp500_curve_path, header=['PortfolioValue'], index=True)
    else: sp500_curve.to_csv(sp500_curve_path, header=['PortfolioValue'], index=False)
    logging.info(f"S&P 500 value curve saved to {sp500_curve_path}")


    logging.info("--- Benchmark Calculation Script Finished ---")

# python .\sp500_calculate_benchmarks.py --config .\configs\sp500_index_config.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate performance metrics for S&P 500 Buy & Hold benchmark.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file containing paths and parameters.")
    args = parser.parse_args()

    benchmark_config = load_config(args.config)
    if benchmark_config:
        main_benchmarks(benchmark_config, args.config)