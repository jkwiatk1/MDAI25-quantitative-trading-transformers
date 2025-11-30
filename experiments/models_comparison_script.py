import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import yaml
import argparse
import logging
import os

def setup_logging(log_file, level=logging.INFO):
    """Configure logging to file and console."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("Comparison script logging setup complete.")

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded comparison configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading comparison config {config_path}: {e}", exc_info=True)
        raise

def load_portfolio_curve(file_path):
    """Load portfolio value curve from CSV."""
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # Ensure column is named 'PortfolioValue' or adapt
        if 'PortfolioValue' not in df.columns:
             # Try to find first numeric column if name is different
             num_cols = df.select_dtypes(include=np.number).columns
             if not num_cols.empty:
                  col_name = num_cols[0]
                  logging.warning(f"Column 'PortfolioValue' not found in {file_path}. Using first numeric column: '{col_name}'.")
                  df.rename(columns={col_name: 'PortfolioValue'}, inplace=True)
             else:
                  raise ValueError(f"No numeric portfolio value column found in {file_path}")

        # Remove 'start' placeholder if it exists
        if 'start' in df.index:
             df = df.drop('start')
        df.index = pd.to_datetime(df.index)
        return df['PortfolioValue']
    except Exception as e:
        logging.error(f"Error loading portfolio curve from {file_path}: {e}")
        return None

def load_benchmark_data(file_path, index_col_name='S&P500', date_col='Date'):
    """Load and process benchmark data."""
    try:
        df = pd.read_excel(file_path)

        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        # Convert index values (handle commas as thousands separators)
        if df[index_col_name].dtype == 'object':
            df[index_col_name] = df[index_col_name].str.replace(',', '', regex=False).astype(float)
        else:
            df[index_col_name] = df[index_col_name].astype(float)

        # Sort by date
        df = df.sort_index()

        # Calculate daily returns
        df['DailyReturn'] = df[index_col_name].pct_change().fillna(0)

        # Calculate cumulative value curve (starting from 1.0)
        df['ValueCurve'] = (1 + df['DailyReturn']).cumprod()
        # Ensure it starts at 1.0 - find first non-zero return
        first_valid_index = df['DailyReturn'].ne(0).idxmax()
        # Shift curve to start at 1 on the day before first return
        start_value = df.loc[:first_valid_index, 'ValueCurve'].iloc[-2]
        df['ValueCurve'] = df['ValueCurve'] / start_value

        # Set first day to 1.0
        df.loc[df.index < first_valid_index, 'ValueCurve'] = 1.0

        return df[['ValueCurve']]
    except Exception as e:
        logging.error(f"Error loading or processing benchmark data from {file_path}: {e}", exc_info=True)
        return None

def main_comparison(config):
    """Main comparison function."""
    results_base_dir = Path(config['results_base_dir'])
    model_names = config['models_to_compare']
    results_suffix = config['results_suffix']
    benchmark_file = config['benchmark_data']['path']
    benchmark_col = config['benchmark_data']['column_name']
    benchmark_date_col = config['benchmark_data']['date_column']
    output_dir_comp = Path(config['comparison_output_dir'])
    output_dir_comp.mkdir(parents=True, exist_ok=True)

    log_file = output_dir_comp / "comparison.log"
    setup_logging(log_file)
    logging.info("--- Starting Comparison Script ---")
    logging.info(f"Comparing models: {model_names}")
    logging.info(f"Results base directory: {results_base_dir}")
    logging.info(f"Comparison output directory: {output_dir_comp}")

    # Load benchmark data
    logging.info(f"Loading benchmark data from: {benchmark_file}")
    df_benchmark_curve = load_benchmark_data(benchmark_file, benchmark_col, benchmark_date_col)
    if df_benchmark_curve is None:
        logging.error("Failed to load benchmark data. Exiting.")
        return

    # Load model results
    model_curves = {}
    min_date = pd.Timestamp.max
    max_date = pd.Timestamp.min

    for model_name in model_names:
        curve_path = results_base_dir / f"{model_name}_GridSearch" / results_suffix / f"{model_name}_portfolio_value_curve.csv"
        if curve_path.exists():
            logging.info(f"Loading results for {model_name} from {curve_path}")
            curve = load_portfolio_curve(curve_path)
            if curve is not None:
                model_curves[model_name] = curve
                # Update date range based on loaded model curves
                min_date = min(min_date, curve.index.min())
                max_date = max(max_date, curve.index.max())
            else:
                logging.warning(f"Could not load curve for {model_name}. Skipping.")
        else:
            logging.warning(f"Curve file not found for {model_name} at {curve_path}. Skipping.")

    if not model_curves:
        logging.error("No model results loaded. Cannot proceed with comparison.")
        return

    logging.info(f"Data loaded for {len(model_curves)} models.")
    logging.info(f"Common analysis period determined by models: {min_date.date()} to {max_date.date()}")

    # Align benchmark to common period
    df_benchmark_test_period = df_benchmark_curve[
        (df_benchmark_curve.index >= min_date) & (df_benchmark_curve.index <= max_date)]
    if df_benchmark_test_period.empty: logging.error("No benchmark data for common period."); return
    # Normalize benchmark to start at 1.0 on min_date
    benchmark_start_value = df_benchmark_test_period.iloc[0]['ValueCurve']
    df_benchmark_curve_aligned = df_benchmark_test_period['ValueCurve'] / benchmark_start_value

    # Generate comparison plot
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 8))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define plot styles for different models
    model_styles = [
        {'color': 'tab:blue', 'linestyle': '-', 'marker': 'o', 'markersize': 3, 'markevery': 50},
        {'color': 'tab:orange', 'linestyle': '--', 'marker': 's', 'markersize': 3, 'markevery': 55},
        {'color': 'tab:green', 'linestyle': ':', 'marker': '^', 'markersize': 3, 'markevery': 60},
        {'color': 'tab:red', 'linestyle': '-.', 'marker': 'd', 'markersize': 3, 'markevery': 65},
        {'color': 'tab:purple', 'linestyle': (0, (3, 1, 1, 1)), 'marker': 'x', 'markersize': 4, 'markevery': 70},
    ]
    benchmark_style = {'color': 'black', 'linestyle': '--', 'linewidth': 2}

    # Plot benchmark
    ax.plot(df_benchmark_curve_aligned.index, df_benchmark_curve_aligned,
            label=f"{benchmark_col} (Buy & Hold)", **benchmark_style)

    # Plot models
    style_idx = 0
    sorted_model_names = sorted(model_curves.keys())
    for model_name in sorted_model_names:
        if model_name == f"{benchmark_col} (B&H)" or model_name == "Equal-Weighted Portfolio": 
            continue

        curve = model_curves[model_name]
        curve_aligned = curve.loc[min_date:max_date]
        curve_normalized = curve_aligned / curve_aligned.iloc[0]

        style = model_styles[style_idx % len(model_styles)]
        ax.plot(curve_normalized.index, curve_normalized,
                label=model_name,
                linewidth=1.5,
                **style)
        style_idx += 1

    # Configure plot settings
    ax.set_title(f"Portfolio Value Comparison ({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})",
                 fontsize=14)
    ax.set_ylabel("Portfolio Value (Normalized to 1.0 at Start)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)

    # Format X-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    fig.autofmt_xdate()

    # Grid
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')

    # Legend
    ax.legend(fontsize=9, loc='upper left')

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save plot with transparent background
    comparison_plot_path = output_dir_comp / "model_vs_benchmark_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight', transparent=True)
    logging.info(f"Comparison plot saved to {comparison_plot_path}")
    plt.close(fig)
    logging.info("--- Comparison Script Finished ---")


# python .\models_comparison_script.py --config .\configs\comparison_config.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare portfolio performance results against benchmarks.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file for the comparison script.")
    args = parser.parse_args()

    comparison_config = load_config(args.config)
    if comparison_config:
        main_comparison(comparison_config)