import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader

from experiments.utils.data_loading import (
    load_finance_data_xlsx,
    prepare_finance_data,
    fill_missing_days,
    get_tickers,
)
from experiments.utils.datasets import (
    MultiStockDataset,
    prepare_sequential_data,
)
from experiments.utils.feature_engineering import calc_input_features
from experiments.utils.metrics import (
    RankLoss,
    calculate_portfolio_performance,
    calculate_predictive_quality,
    calculate_precision_at_k
)
from experiments.utils.training import (
    evaluate_model,
    inverse_transform_predictions,
)

from models.PortfolioVanillaTransformer import build_PortfolioVanillaTransformer
from models.PortfolioCrossFormer import build_CrossFormer
from models.PortfolioMASTER import build_MASTER
from models.PortfolioiTransformer import build_PortfolioITransformer
from models.PortfolioTransformerCA import build_TransformerCA


# drop_tickers_list = ['CEG', 'GEV']
OUTPUT_FOLDER_SUFFIX = "_1011"

def setup_logging(log_file):
    """Configure logging to file and console."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
        # mode='w' nadpisuje log przy każdym uruchomieniu
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("Logging setup complete.")


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading config {config_path}: {e}")
        raise



def main(config_path: str):
    try:
        config = load_config(config_path)
    except Exception:
        return

    # --- Setup ---
    model_name = config["model"]["name"]
    output_base_dir = Path(config["data"].get("output_dir", "results"))
    output_dir = output_base_dir / f"{model_name}_GridSearch" / f"Evaluation_{config['portfolio']['top_k']}_best_assets{OUTPUT_FOLDER_SUFFIX}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "evaluation.log"
    setup_logging(log_file)
    logging.info(f"Starting evaluation for model: {model_name}")
    logging.info(f"Configuration loaded from: {config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and config["training"].get("use_cuda", True) else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Output directory: {output_dir}")

    start_date = pd.to_datetime(config["data"]["start_date"])
    end_date = pd.to_datetime(config["data"]["end_date"])
    lookback = config["training"]["lookback"]
    train_split_ratio = 1.0 - config["training"]["test_split"]
    evaluation_split_ratio = config["training"]["test_split"]
    logging.info(
        f"Using Train ratio: {train_split_ratio:.2f}, Evaluation (Val+Test) ratio: {evaluation_split_ratio:.2f}")
    if train_split_ratio <= 0 or train_split_ratio >= 1:
        raise ValueError("Resulting train_split_ratio must be between 0 and 1.")

    try:
        selected_tickers = get_tickers(config)
        # selected_tickers = [ticker for ticker in selected_tickers if ticker not in drop_tickers_list]

        stock_amount = len(selected_tickers)
        if stock_amount == 0: raise ValueError("No tickers specified or loaded.")
        logging.info(f"Processing {stock_amount} tickers.")
    except Exception as e:
        logging.error(f"Error getting tickers: {e}", exc_info=True)
        return

    logging.info("--- Loading and Preparing Data ---")
    try:
        data_raw_dict, _ = load_finance_data_xlsx(
            config["data"]["path"], config["data"].get("yahoo_data", False)
        )
        data_filled_dict = fill_missing_days(data_raw_dict, selected_tickers, start_date, end_date)
        original_data_with_dates = prepare_finance_data(data_filled_dict, selected_tickers,
                                                        config["data"]["init_cols_to_use"] + ["Date"])
        first_ticker = selected_tickers[0]
        if first_ticker not in original_data_with_dates:
            raise ValueError(f"Data for first ticker '{first_ticker}' not found after preparation.")
        all_original_dates = original_data_with_dates[first_ticker].Date.copy()

        data_features = calc_input_features(original_data_with_dates, selected_tickers,
                                            config["data"]["preproc_cols_to_use"], lookback)
        preproc_cols = config["data"].get("preproc_cols_to_use", [])
        if not preproc_cols: raise ValueError("`preproc_cols_to_use` cannot be empty.")

        financial_features_amount = len(preproc_cols)
        data_final_features = {key: value[preproc_cols] for key, value in data_features.items() if key in data_features}

        train_data_dict = {}
        eval_data_dict = {}
        eval_data_dict_orginal = {}
        for ticker in selected_tickers:
            df_ticker = data_final_features[ticker]
            n_samples_ticker = len(df_ticker)
            train_end_idx_ticker = int(train_split_ratio * n_samples_ticker)
            train_data_dict[ticker] = df_ticker.iloc[:train_end_idx_ticker]
            eval_data_dict[ticker] = df_ticker.iloc[train_end_idx_ticker:]
            eval_data_dict_orginal[ticker] = df_ticker.iloc[train_end_idx_ticker:]
        logging.info(
            f"Data split: Train size={len(train_data_dict[first_ticker])}, Eval size={len(eval_data_dict[first_ticker])}")

        scalers = {ticker: {} for ticker in selected_tickers}
        eval_data_scaled_dict = {}
        
        for ticker in selected_tickers:
            if ticker not in eval_data_dict: continue
            eval_df = eval_data_dict[ticker].copy()
            eval_data_scaled_dict[ticker] = pd.DataFrame(index=eval_df.index)
            
            for feature in preproc_cols:
                if feature not in eval_df.columns:
                    logging.warning(f"Feature '{feature}' missing for ticker '{ticker}'. Skipping.")
                    continue
                scaler = MinMaxScaler(feature_range=(0, 1))
                try:
                    # For evaluation, fit on available eval data (or load pre-fitted scalers)
                    scaler.fit(train_data_dict[ticker][[feature]])
                    eval_scaled_values = scaler.transform(eval_df[[feature]])
                    eval_data_scaled_dict[ticker][feature] = eval_scaled_values.flatten()
                    scalers[ticker][feature] = scaler
                except Exception as e:
                    logging.error(f"Error scaling '{feature}' for '{ticker}': {e}")
                    eval_data_scaled_dict[ticker][feature] = eval_df[feature].values
                    scalers[ticker][feature] = None
        fitted_scalers = scalers

        # Save fitted scalers
        scalers_path = output_dir / "fitted_scalers_eval.joblib"
        joblib.dump(fitted_scalers, scalers_path)
        logging.info("Saved scalers.")

        # --- Prepare sequence on for evaluation set ---
        target_col_name = config["data"].get("preproc_target_col", preproc_cols[0])
        target_col_index = preproc_cols.index(target_col_name)
        eval_sequences, eval_targets_scaled, _ = prepare_sequential_data(
            eval_data_scaled_dict, selected_tickers, lookback, target_col_index
        )
        _, eval_targets_orginal, _ = prepare_sequential_data(
            eval_data_dict_orginal, selected_tickers, lookback, target_col_index
        )
        logging.info(
            f"Evaluation sequences prepared. Seq shape: {eval_sequences.shape}, Tgt shape: {eval_targets_scaled.shape}")
        if len(eval_sequences) == 0: raise ValueError("Evaluation sequences are empty.")

        # Get proper dates for evaluation period
        num_total_days = len(all_original_dates)
        eval_start_original_idx_pos = int(train_split_ratio * num_total_days) + lookback
        last_eval_target_original_idx_pos = len(all_original_dates) - 1
        if eval_start_original_idx_pos >= len(all_original_dates): raise ValueError("Eval start index out of bounds.")

        actual_eval_dates = all_original_dates[eval_start_original_idx_pos: last_eval_target_original_idx_pos + 1]

        if len(actual_eval_dates) != len(eval_sequences):
            logging.warning(
                f"Length mismatch: actual_eval_dates ({len(actual_eval_dates)}) vs eval_sequences ({len(eval_sequences)}). Using sequence length for dates.")
            actual_eval_dates = actual_eval_dates[:len(eval_sequences)]
            if len(actual_eval_dates) != len(eval_sequences):
                logging.error("Cannot align dates with evaluation sequences. Plotting without dates.")
                actual_eval_dates = None

        if actual_eval_dates is not None:
            logging.info(
                f"Evaluation period dates range from {actual_eval_dates.min().date()} to {actual_eval_dates.max().date()}")

        eval_loader = DataLoader(
            MultiStockDataset(eval_sequences, eval_targets_scaled),
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"].get("num_workers", 0),
            pin_memory=device.type == 'cuda'
        )

    except Exception as e:
        logging.error(f"Error during data loading/preparation for eval: {e}", exc_info=True)
        return

    logging.info(f"--- Building Model: {model_name} ---")
    model_params = config["model"]

    try:
        common_params_build = {
            "stock_amount": stock_amount,
            "financial_features_amount": financial_features_amount,
            "lookback": lookback,
            "device": device,
        }
        builder_args = {**common_params_build, **model_params}
        builder_args.pop('name', None)

        if model_name == "VanillaTransformer":
            model = build_PortfolioVanillaTransformer(
                stock_amount=stock_amount,
                financial_features_amount=len(config["data"]["preproc_cols_to_use"]),
                lookback=config["training"]["lookback"],
                d_model=config["model"]["d_model"],
                n_heads=config["model"]["n_heads"],
                d_ff=config["model"]["d_ff"],
                dropout=config["model"]["dropout"],
                num_encoder_layers=config["model"]["num_encoder_layers"],
                device=device,
            ).to(device)
        elif model_name == "CrossFormer":
            model = build_CrossFormer(
                stock_amount=stock_amount,
                financial_features=len(config["data"]["preproc_cols_to_use"]),
                in_len=config["training"]["lookback"],
                seg_len=config["model"]["seg_len"],
                win_size=config["model"]["win_size"],
                factor=config["model"]["factor"],
                aggregation_type="avg_pool",
                d_model=config["model"]["d_model"],
                d_ff=config["model"]["d_ff"],
                n_heads=config["model"]["n_heads"],
                # Użyj num_encoder_layers jako e_layers dla spójności
                e_layers=config["model"]["num_encoder_layers"],
                dropout=config["model"]["dropout"],
                device=device,
            )
        elif model_name == "MASTER":
            model = build_MASTER(
                stock_amount=stock_amount,
                financial_features_amount=len(config["data"]["preproc_cols_to_use"]),
                lookback=config["training"]["lookback"],
                d_model=config["model"]["d_model"],
                n_heads=config["model"]["n_heads"],
                d_ff=config["model"]["d_ff"],
                dropout=config["model"]["dropout"],
                num_encoder_layers=config["model"]["num_encoder_layers"],
                device=device,
            ).to(device)
        elif model_name == "iTransformer":
            model = build_PortfolioITransformer(
                stock_amount=stock_amount,
                financial_features_amount=len(config["data"]["preproc_cols_to_use"]),
                lookback=config["training"]["lookback"],
                d_model=config["model"]["d_model"],
                n_heads=config["model"]["n_heads"],
                d_ff=config["model"]["d_ff"],
                dropout=config["model"]["dropout"],
                num_encoder_layers=config["model"]["num_encoder_layers"],
                device=device,
            ).to(device)
        elif model_name == "TransformerCA":
            model = build_TransformerCA(
                stock_amount=stock_amount,
                financial_features_amount=len(config["data"]["preproc_cols_to_use"]),
                lookback=config["training"]["lookback"],
                d_model=config["model"]["d_model"],
                n_heads=config["model"]["n_heads"],
                d_ff=config["model"]["d_ff"],
                dropout=config["model"]["dropout"],
                num_encoder_layers=config["model"]["num_encoder_layers"],
                device=device,
            ).to(device)
        else:
            logging.error(f"Unknown model name in config: {model_name}")
            return

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Built {model_name} with {total_params:,} trainable parameters.")

    except KeyError as e:
        logging.error(f"Missing key in config['model'] for {model_name}: {e}. Check config file.")
        return
    except Exception as e:
        logging.error(f"Error building model {model_name}: {e}", exc_info=True)
        return

    weights_path = config["model"].get("weights_path")
    if not weights_path or not Path(weights_path).exists():
        logging.error(f"Model weights path '{weights_path}' not found or not specified in config.")
        return
    try:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.to(device)
        logging.info(f"Successfully loaded model weights from {weights_path}")
    except Exception as e:
        logging.error(f"Error loading model weights from {weights_path}: {e}", exc_info=True)
        return

    logging.info("--- Evaluating Model on Combined Val+Test Set ---")
    try:
        eval_loss_type = config["training"].get("loss_function", "RankLoss")
        if eval_loss_type == "RankLoss":
            criterion = RankLoss(lambda_rank=config["training"].get("lambda_rank", 0.5))
        elif eval_loss_type == "MSE":
            criterion = torch.nn.MSELoss()
        else:
            logging.warning(f"Using MSE loss for evaluation as '{eval_loss_type}' is not MSE or RankLoss.")
            criterion = torch.nn.MSELoss()

        use_amp_eval = config["training"].get("use_amp", True) and device.type == 'cuda'
        scaler_eval = torch.cuda.amp.GradScaler(enabled=use_amp_eval)

        predictions_scaled, targets_scaled_eval, eval_loss = evaluate_model(
            model=model,
            test_loader=eval_loader,
            criterion=criterion,
            device=device,
            scaler=scaler_eval
        )
        logging.info(f"Evaluation finished. Eval Loss ({criterion.__class__.__name__}): {eval_loss:.6f}")

        if predictions_scaled.shape[-1] == 1:
            predictions_scaled = predictions_scaled.squeeze(-1)
        if targets_scaled_eval.shape[-1] == 1:
            targets_scaled_eval = targets_scaled_eval.squeeze(-1)

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}", exc_info=True)
        return

    logging.info("--- Calculating Metrics on Combined Val+Test Set ---")
    try:
        predictive_metrics = calculate_predictive_quality(predictions_scaled, targets_scaled_eval)

        _, targets_inv = inverse_transform_predictions(predictions_scaled, targets_scaled_eval, selected_tickers, fitted_scalers, target_col_name)

        targets_orginal = eval_targets_orginal.squeeze(-1).numpy()

        portfolio_top_k = config["portfolio"].get("top_k", 5)
        precision_at_k_value = calculate_precision_at_k(predictions_scaled, targets_orginal, top_k=portfolio_top_k)
        predictive_metrics[f'Precision@{portfolio_top_k}'] = precision_at_k_value

        portfolio_risk_free_rate = config["portfolio"].get("risk_free_rate", 0.043)
        portfolio_metrics, portfolio_value_curve = calculate_portfolio_performance(
            predictions_scaled, targets_orginal, top_k=portfolio_top_k, risk_free_rate=portfolio_risk_free_rate
        )

        all_metrics = {**portfolio_metrics, **predictive_metrics, f"Eval Loss ({eval_loss_type})": eval_loss}
        logging.info(f"Final Combined Metrics: {all_metrics}")

    except Exception as e:
        logging.error(f"Error calculating metrics: {e}", exc_info=True)
        return

    try:
        results_file = output_dir / "evaluation_results.txt"
        csv_results_path = output_dir / "evaluation_results.csv"

        with open(results_file, "w") as f:
            f.write(f"Evaluation Results for Model: {model_name}\n")
            f.write(f"Config Path: {config_path}\n")
            f.write(f"Weights Path: {weights_path}\n")
            f.write(f"Number of Stocks: {stock_amount}\n")
            f.write(f"Features Used: {preproc_cols}\n")
            f.write(f"Lookback: {lookback}\n")
            f.write(f"Evaluation Set Size: {len(eval_sequences)}\n")
            f.write("-" * 30 + "\nMETRICS:\n")
            for key, value in all_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        logging.info(f"Saved evaluation results to {results_file}")

        pd.DataFrame([all_metrics]).to_csv(csv_results_path, index=False)
        logging.info(f"Saved evaluation metrics to CSV: {csv_results_path}")

        curve_df = pd.DataFrame({'PortfolioValue': portfolio_value_curve})
        start_curve_original_idx_pos = eval_start_original_idx_pos - 1
        if start_curve_original_idx_pos >= 0:
            curve_dates = all_original_dates[start_curve_original_idx_pos: last_eval_target_original_idx_pos + 1]
            if len(curve_dates) == len(portfolio_value_curve):
                curve_df.index = pd.to_datetime(curve_dates)
                logging.info(f"Portfolio curve dates set from {curve_dates.min().date()} to {curve_dates.max().date()}")
            else:
                logging.warning(
                    f"Length mismatch for curve dates ({len(curve_dates)}) vs curve values ({len(portfolio_value_curve)}). Saving without date index.")
        else:
            logging.warning("Cannot determine start date for curve index. Saving without date index.")
        curve_path = output_dir / f"{model_name}_portfolio_value_curve.csv"
        curve_df.to_csv(curve_path, index=isinstance(curve_df.index, pd.DatetimeIndex))
        logging.info(f"Saved portfolio value curve data to {curve_path}")

        comp_output_dir = output_base_dir / f"evaluation_final_Top{portfolio_top_k}" / model_name
        comp_output_dir.mkdir(parents=True, exist_ok=True)
        comp_curve_path = comp_output_dir / f"{model_name}_portfolio_value_curve.csv"
        curve_df.to_csv(comp_curve_path, index=isinstance(curve_df.index, pd.DatetimeIndex))
        logging.info(f"Saved portfolio value curve data to comparison dir: {comp_curve_path}")

    except Exception as e:
        logging.error(f"Error saving results: {e}", exc_info=True)

    try:
        plt.figure(figsize=(12, 6))
        if isinstance(curve_df.index, pd.DatetimeIndex):
            plt.plot(curve_df.index, curve_df['PortfolioValue'], linestyle="-", label=f"{model_name} Portfolio Value")
            plt.xlabel("Date")
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
            plt.gcf().autofmt_xdate()
        else:
            # Fallback if no dates available
            plt.plot(np.arange(len(portfolio_value_curve)), portfolio_value_curve, linestyle="-",
                     label=f"{model_name} Portfolio Value")
            plt.xlabel(f"Time Points ({len(portfolio_value_curve)} points)")

        plt.ylabel("Portfolio Value (Starts at 1.0)")
        plt.title(f"Portfolio Value Over Time ({model_name}, Top-{portfolio_top_k} Strategy)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which='both', linestyle=':')
        plt.tight_layout()
        plot_path = output_dir / f"{model_name}_portfolio_value_curve.png"
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Saved portfolio value plot to {plot_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error during plotting: {e}", exc_info=True)

    logging.info(f"--- Evaluation script finished for {model_name} ---")


if __name__ == "__main__":

    models_to_run = [
        {
            "model_name": "VanillaTransformer",
            "config_file_path": "../experiments/configs/test_config_VanillaTransformer.yaml"
        },
        {
            "model_name": "TransformerCA",
            "config_file_path": "../experiments/configs/test_config_TransformerCA.yaml"
        },
        {
            "model_name": "iTransformer",
            "config_file_path": "../experiments/configs/test_config_iTransformer.yaml"
        },
        {
            "model_name": "CrossFormer",
            "config_file_path": "../experiments/configs/test_config_CrossFormer.yaml"
        },
        {
            "model_name": "MASTER",
            "config_file_path": "../experiments/configs/test_config_MASTER.yaml"
        }
    ]

    for model in models_to_run:
        model_name_to_run = model["model_name"]
        config_file_path = model["config_file_path"]

        print(f"\n--- Running Evaluation for: {model_name_to_run} ---")
        print(f"--- Using Config: {config_file_path} ---\n")

        if not Path(config_file_path).exists():
            print(f"ERROR: Configuration file not found at {config_file_path}")
        else:
            try:
                main(config_path=config_file_path)
            except Exception as e:
                logging.error(f"Critical error during main execution for {model_name_to_run}: {e}", exc_info=True)
                print(f"CRITICAL ERROR running {model_name_to_run}. Check logs for details.")

        print(f"\n--- Finished Evaluation for: {model_name_to_run} ---")
