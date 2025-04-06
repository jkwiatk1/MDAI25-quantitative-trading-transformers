import logging
import os
from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
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
    normalize_data,
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


def setup_logging(log_file):
    """Konfiguruje logowanie do pliku i konsoli."""
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
    """Wczytuje konfigurację z pliku YAML."""
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
    output_dir = output_base_dir / f"{model_name}_GridSearch" / f"Evaluation_{config['portfolio']['top_k']}_best_assets"
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
    test_split_ratio = config["training"]["test_split"]

    try:
        selected_tickers = get_tickers(config)
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
        original_data_with_dates = prepare_finance_data(data_filled_dict, selected_tickers, config["data"]["init_cols_to_use"]+ ["Date"])

        first_ticker = selected_tickers[0]
        if first_ticker not in original_data_with_dates:
            raise ValueError(f"Data for first ticker '{first_ticker}' not found after preparation.")
        all_original_dates = original_data_with_dates[first_ticker].Date.copy()

        data_processed = calc_input_features(original_data_with_dates, selected_tickers,
                                             config["data"]["preproc_cols_to_use"], lookback)
        preproc_cols = config["data"].get("preproc_cols_to_use", [])
        if not preproc_cols: raise ValueError("`preproc_cols_to_use` cannot be empty.")
        financial_features_amount = len(preproc_cols)
        data_final_features = {key: value[preproc_cols] for key, value in data_processed.items() if key in data_processed}

        data_scaled, feat_scalers = normalize_data(data_final_features, selected_tickers, preproc_cols)
        target_col_name = config["data"].get("preproc_target_col", preproc_cols[0])
        target_col_index = preproc_cols.index(target_col_name)
        logging.info(f"Using '{target_col_name}' (index {target_col_index}) as target variable.")

        sequences, targets, _ = prepare_sequential_data(data_scaled, selected_tickers, lookback, target_col_index)
        logging.info(f"Data sequences prepared. Seq shape: {sequences.shape}, Tgt shape: {targets.shape}")

        # --- Prepare evaluation data ---
        num_sequences_total = len(sequences)
        eval_start_index = int((1.0 - test_split_ratio) * num_sequences_total)

        test_sequences = sequences[eval_start_index:]
        test_targets = targets[eval_start_index:]

        if len(test_sequences) == 0:
            raise ValueError("Test set is empty after splitting.")
        logging.info(f"Evaluation set size (sequences): {len(test_sequences)}")

        first_test_target_original_idx_pos = eval_start_index + lookback
        last_test_target_original_idx_pos = num_sequences_total - 1 + lookback

        if last_test_target_original_idx_pos >= len(all_original_dates):
            logging.warning(f"Calculated last test date index ({last_test_target_original_idx_pos}) exceeds "
                            f"original data length ({len(all_original_dates)}). Adjusting.")
            last_test_target_original_idx_pos = len(all_original_dates) - 1

        if first_test_target_original_idx_pos > last_test_target_original_idx_pos:
            raise ValueError("Cannot determine test date range due to index mismatch.")

        logging.info(
            f"Calculated indices for slicing: start={first_test_target_original_idx_pos}, stop={last_test_target_original_idx_pos + 1}")
        logging.info(f"Length of all_original_dates: {len(all_original_dates)}")
        # Sprawdź, czy indeksy są w granicach
        if first_test_target_original_idx_pos < 0 or last_test_target_original_idx_pos >= len(
                all_original_dates) or first_test_target_original_idx_pos > last_test_target_original_idx_pos:
            logging.error("Invalid date index range calculated!")

        actual_test_dates = all_original_dates[
                            first_test_target_original_idx_pos: last_test_target_original_idx_pos + 1]
        logging.info(f"Type of all_original_dates: {type(all_original_dates)}")
        if isinstance(all_original_dates, pd.DatetimeIndex):
            logging.info(
                f"Sample dates from all_original_dates: {all_original_dates[:5].tolist()}")  # Pokaż pierwsze 5 dat
        else:
            logging.error("all_original_dates is NOT a DatetimeIndex!")

        logging.info(f"Length of actual_test_dates: {len(actual_test_dates)}")
        logging.info(f"Type of actual_test_dates: {type(actual_test_dates)}")
        if len(actual_test_dates) > 0 and isinstance(actual_test_dates, pd.DatetimeIndex):
            logging.info(f"Min test date: {actual_test_dates.min()}, Max test date: {actual_test_dates.max()}")
            # Spróbuj wywołać .date() tutaj, aby zobaczyć, czy działa
            try:
                logging.info(
                    f"Min test date object: {actual_test_dates.min()} with date attribute: {actual_test_dates.min().date()}")
            except AttributeError as e:
                logging.error(f"AttributeError on min date: {e}")
        elif len(actual_test_dates) == 0:
            logging.error("actual_test_dates is empty!")
        else:
            logging.error(f"actual_test_dates is not a DatetimeIndex, type is {type(actual_test_dates)}")

        logging.info(
            f"Test period dates range from {actual_test_dates.min().date()} to {actual_test_dates.max().date()}")

        test_loader = DataLoader(
            MultiStockDataset(test_sequences, test_targets),
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"].get("num_workers", 0),
            pin_memory=device.type == 'cuda'  # Pin memory tylko dla GPU
        )
        # first_ticker = selected_tickers[0]
        # all_dates = data[first_ticker].index if first_ticker in data else None
        # test_dates = all_dates if all_dates is not None else None
        # if test_dates is not None and len(test_dates) != len(test_sequences):
        #     logging.warning(
        #         f"Length mismatch between expected test dates ({len(test_dates)}) and actual test sequences ({len(test_sequences)}). Plotting without dates.")
        #     test_dates = None

    except Exception as e:
        logging.error(f"Error during data loading/preparation: {e}", exc_info=True)
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

    logging.info("--- Evaluating Model ---")
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

        predictions_scaled, targets_scaled, test_loss = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            scaler=scaler_eval
        )
        logging.info(f"Evaluation finished. Test Loss ({criterion.__class__.__name__}): {test_loss:.6f}")

        if predictions_scaled.shape[-1] == 1:
            predictions_scaled = predictions_scaled.squeeze(-1)
        if targets_scaled.shape[-1] == 1:
            targets_scaled = targets_scaled.squeeze(-1)

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}", exc_info=True)
        return

    logging.info("--- Calculating Metrics ---")
    try:
        # Metryki predykcyjne (na danych przeskalowanych)
        predictive_metrics = calculate_predictive_quality(predictions_scaled, targets_scaled)

        # Odwrócenie transformacji tylko dla targetów (do metryk portfelowych)
        _, targets_inv = inverse_transform_predictions(
            predictions_scaled,
            targets_scaled,
            selected_tickers,
            feat_scalers,
            target_col_name,
        )

        # Precision@k (używa odwróconych targetów)
        portfolio_top_k = config["portfolio"].get("top_k", 5)  # Pobierz k z sekcji portfolio configu
        precision_at_k_value = calculate_precision_at_k(predictions_scaled, targets_inv, top_k=portfolio_top_k)
        predictive_metrics[f'Precision@{portfolio_top_k}'] = precision_at_k_value

        # Metryki portfelowe (używają oryginalnych predykcji i odwróconych targetów)
        portfolio_risk_free_rate = config["portfolio"].get("risk_free_rate", 0.0)  # Pobierz Rf z sekcji portfolio
        portfolio_metrics, portfolio_value_curve = calculate_portfolio_performance(
            predictions_scaled, targets_inv, top_k=portfolio_top_k, risk_free_rate=portfolio_risk_free_rate
        )

        all_metrics = {**portfolio_metrics, **predictive_metrics, f"Test Loss ({eval_loss_type})": test_loss}
        logging.info(f"Final Combined Metrics: {all_metrics}")

    except Exception as e:
        logging.error(f"Error calculating metrics: {e}", exc_info=True)
        return

    try:
        results_file = output_dir / "evaluation_results.txt"
        with open(results_file, "w") as f:
            f.write(f"Evaluation Results for Model: {model_name}\n")
            f.write(f"Config Path: {config_path}\n")
            f.write(f"Weights Path: {weights_path}\n")
            f.write(f"Number of Stocks: {stock_amount}\n")
            f.write(f"Features Used: {preproc_cols}\n")
            f.write(f"Lookback: {lookback}\n")
            f.write(f"Test Set Size: {len(test_sequences)}\n")
            f.write("-" * 30 + "\nMETRICS:\n")
            for key, value in all_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        logging.info(f"Saved evaluation results to {results_file}")

        csv_results_path = output_dir / "evaluation_results.csv"
        pd.DataFrame([all_metrics]).to_csv(csv_results_path, index=False)
        logging.info(f"Saved evaluation metrics to CSV: {csv_results_path}")

        curve_df = pd.DataFrame({'PortfolioValue': portfolio_value_curve})
        start_curve_original_idx_pos = first_test_target_original_idx_pos - 1
        if start_curve_original_idx_pos >= 0:
            curve_dates = all_original_dates[start_curve_original_idx_pos: last_test_target_original_idx_pos + 1]
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


        # if test_dates is not None and len(test_dates) == len(portfolio_value_curve) - 1:  # Curve ma +1 element (start)
        #     curve_df.index = pd.to_datetime(['start'] + list(test_dates))
        # curve_path = output_dir / f"{model_name}_portfolio_value_curve.csv"
        # curve_path_for_models_comparison = output_base_dir / f"evaluation_final_{config['portfolio']['top_k']}_best_assets"
        # curve_path_for_models_comparison.mkdir(parents=True, exist_ok=True)
        # curve_df.to_csv(curve_path)
        # curve_df.to_csv(curve_path_for_models_comparison/f"{model_name}_portfolio_value_curve.csv")
        # logging.info(f"Saved portfolio value curve data to {curve_path} & {curve_path_for_models_comparison}")

    except Exception as e:
        logging.error(f"Error saving results: {e}", exc_info=True)

    try:
        plt.figure(figsize=(12, 6))
        # --- **Poprawione Rysowanie z DATAMI** ---
        if isinstance(curve_df.index, pd.DatetimeIndex):
            plt.plot(curve_df.index, curve_df['PortfolioValue'], linestyle="-", label=f"{model_name} Portfolio Value")
            plt.xlabel("Date")
            # Lepsze formatowanie osi X
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
            plt.gcf().autofmt_xdate()  # Automatyczne obracanie etykiet
        else:
            # Fallback, jeśli nie ma dat
            plt.plot(np.arange(len(portfolio_value_curve)), portfolio_value_curve, linestyle="-",
                     label=f"{model_name} Portfolio Value")
            plt.xlabel(f"Time Points (Test Period + Start, {len(portfolio_value_curve)} points)")

        plt.ylabel("Portfolio Value (Starts at 1.0)")
        plt.title(f"Portfolio Value Over Time ({model_name}, Top-{portfolio_top_k} Strategy)")
        plt.yscale('log')  # Dodana skala logarytmiczna dla lepszej wizualizacji wzrostu
        plt.legend()
        plt.grid(True, which='both', linestyle=':')  # Poprawiony grid
        plt.tight_layout()
        plot_path = output_dir / f"{model_name}_portfolio_value_curve.png"
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Saved portfolio value plot to {plot_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error during plotting: {e}", exc_info=True)

    logging.info(f"--- Evaluation script finished for {model_name} ---")


    # try:
    #     plt.figure(figsize=(12, 6))
    #     if test_dates is not None and len(test_dates) == len(portfolio_value_curve) - 1:
    #         plot_dates = pd.to_datetime(list(test_dates))
    #         plt.plot(plot_dates, portfolio_value_curve[1:], linestyle="-",
    #                  label=f"{model_name} Portfolio Value")
    #         plt.xlabel("Date")
    #         plt.xticks(rotation=45)
    #     else:
    #         plt.plot(portfolio_value_curve[1:], linestyle="-",
    #                  label=f"{model_name} Portfolio Value")
    #         plt.xlabel(f"Trading Days (Test Period, {len(portfolio_value_curve) - 1} days)")
    #
    #     plt.ylabel("Portfolio Value (Starts at 1.0)")
    #     plt.title(f"Portfolio Value Over Time ({model_name}, Top-{portfolio_top_k} Strategy)")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plot_path = output_dir / f"{model_name}_portfolio_value_curve.png"
    #     plt.savefig(plot_path, dpi=300)
    #     logging.info(f"Saved portfolio value plot to {plot_path}")
    #     # plt.show() # Odkomentuj, jeśli chcesz pokazywać wykresy interaktywnie
    #     plt.close()  # Zamknij figurę po zapisaniu
    #
    # except Exception as e:
    #     logging.error(f"Error during plotting: {e}", exc_info=True)
    #
    # logging.info(f"--- Evaluation script finished for {model_name} ---")


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

    # Iteracja po liście modeli
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

# if __name__ == "__main__":
#     # Użyj argparse do przekazywania ścieżki do configu
#     parser = argparse.ArgumentParser(description="Evaluate trained model and compute portfolio metrics.")
#     parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file for the model evaluation.")
#     args = parser.parse_args()
#
#     main(args.config) # Przekaż ścieżkę do configu do funkcji main
#
#     # Przykładowe wywołanie z linii komend:
#     # python your_evaluation_script_name.py --config experiments/configs/test_config_VanillaTransformer.yaml
#     # python your_evaluation_script_name.py --config experiments/configs/test_config_CrossFormer.yaml
#     # ... itd.
