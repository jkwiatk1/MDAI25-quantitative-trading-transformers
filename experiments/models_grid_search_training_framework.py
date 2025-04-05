import logging
import os
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
from types import SimpleNamespace
import itertools  # Potrzebne do generowania kombinacji
import argparse
import time  # Do mierzenia czasu
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
from types import SimpleNamespace

from experiments.utils.data_loading import (
    fill_missing_days,
    load_finance_data_xlsx,
    prepare_finance_data,
    get_tickers,
)
from experiments.utils.datasets import (
    prepare_sequential_data,
    normalize_data,
    MultiStockDataset,
)
from experiments.utils.feature_engineering import calc_input_features
from experiments.utils.metrics import RankLoss
from experiments.utils.training import (
    train_model,
    evaluate_model,
    inverse_transform_predictions,
    plot_predictions,
)
from models.PortfolioTransformerCA import build_TransformerCA
from models.PortfolioCrossFormer import build_CrossFormer
from models.PortfolioMASTER import build_MASTER
from models.PortfolioiTransformer import build_PortfolioITransformer
from models.PortfolioVanillaTransformer import build_PortfolioVanillaTransformer


def setup_logging(log_file, level=logging.INFO):
    """Konfiguruje logowanie do pliku i konsoli."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
        datefmt="%Y-%m-%d %H:%M:%S",
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


def build_model_dynamically(model_name, config, common_params):
    builder_args_all = {**common_params, **config}
    builder_args = {
        k: v for k, v in builder_args_all.items() if k not in ["name", "weights_path"]
    }

    logging.debug(f"Attempting to build {model_name} with args: {builder_args}")

    if model_name == "VanillaTransformer":
        model = build_PortfolioVanillaTransformer(
            stock_amount=common_params["stock_amount"],
            financial_features_amount=common_params["financial_features_amount"],
            lookback=config["lookback"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            dropout=config.get("dropout", 0.1),
            num_encoder_layers=config["num_encoder_layers"],
            device=common_params['device'],
        ).to(common_params['device'])
    elif model_name == "CrossFormer":
        model = build_CrossFormer(
            stock_amount=common_params["stock_amount"],
            financial_features=common_params["financial_features_amount"],
            in_len=config["lookback"],
            seg_len=config["seg_len"],
            win_size=config["win_size"],
            factor=config["factor"],
            aggregation_type=config.get(
                "aggregation_type", "avg_pool"
            ),  # TODO add to config
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            n_heads=config["n_heads"],
            e_layers=config["num_encoder_layers"],
            dropout=config["dropout"],
            device=common_params['device'],
        ).to(common_params['device'])
    elif model_name == "MASTER":
        model = build_MASTER(
            stock_amount=common_params["stock_amount"],
            financial_features_amount=common_params["financial_features_amount"],
            lookback=config["lookback"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            num_encoder_layers=config["num_encoder_layers"],
            device=common_params['device'],
        ).to(common_params['device'])
    elif model_name == "iTransformer":
        model = build_PortfolioITransformer(
            stock_amount=common_params["stock_amount"],
            financial_features_amount=common_params["financial_features_amount"],
            lookback=config["lookback"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            num_encoder_layers=config["num_encoder_layers"],
            device=common_params['device'],
        ).to(common_params['device'])
    elif model_name == "TransformerCA":
        model = build_TransformerCA(
            stock_amount=common_params["stock_amount"],
            financial_features_amount=common_params["financial_features_amount"],
            lookback=config["lookback"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            num_encoder_layers=config["num_encoder_layers"],
            device=common_params['device']
        ).to(common_params['device'])
    else:
        raise ValueError(f"Unknown model name in config: {model_name}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.debug(f"Built {model_name} with {total_params:,} trainable parameters.")
    return model


def run_grid_search(config_path: str):
    try:
        config = load_config(config_path)
    except Exception:
        return

    model_base_name = config["model"]["name"]
    output_base_dir = (
            Path(config["data"].get("output_dir", "results"))
            / f"{model_base_name}_GridSearch"
    )
    output_base_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_base_dir / "grid_search.log"
    setup_logging(log_file)
    logging.info(f"Starting Grid Search for model base: {model_base_name}")
    logging.info(f"Configuration loaded from: {config_path}")
    logging.info(f"Output directory for grid search: {output_base_dir}")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and config["training"].get("use_cuda", True)
        else "cpu"
    )
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("--- Loading and Preparing Data (Once) ---")
    try:
        start_date = pd.to_datetime(config["data"]["start_date"])
        end_date = pd.to_datetime(config["data"]["end_date"])
        lookback = config["training"]["lookback"]
        selected_tickers = get_tickers(config)
        stock_amount = len(selected_tickers)
        if stock_amount == 0:
            raise ValueError("No tickers found.")
        logging.info(f"Processing {stock_amount} tickers.")

        preproc_cols = config["data"].get("preproc_cols_to_use", [])
        if not isinstance(preproc_cols, list) or not preproc_cols:
            raise ValueError("`preproc_cols_to_use` must be a non-empty list.")
        financial_features_amount = len(preproc_cols)

        data_raw_dict, _ = load_finance_data_xlsx(
            config["data"]["path"], config["data"].get("yahoo_data", False)
        )
        data_filled_dict = fill_missing_days(
            data_raw_dict, selected_tickers, start_date, end_date
        )
        data = prepare_finance_data(
            data_filled_dict, selected_tickers, config["data"]["init_cols_to_use"]
        )
        data = calc_input_features(data, selected_tickers, preproc_cols, lookback)
        data = {key: value[preproc_cols] for key, value in data.items() if key in data}
        data_scaled, _ = normalize_data(
            data, selected_tickers, preproc_cols
        )  # Scalery nie są potrzebne tutaj

        target_col_name = config["data"].get("preproc_target_col", preproc_cols[0])
        target_col_index = preproc_cols.index(target_col_name)
        sequences, targets, _ = prepare_sequential_data(
            data_scaled, selected_tickers, lookback, target_col_index
        )

        # --- Data Splitting ---
        logging.info("--- Splitting Data ---")
        test_split_ratio = config["training"]["test_split"]
        val_split_ratio = config["training"]["val_split"]
        num_samples = len(sequences)
        train_size_abs = int((1 - test_split_ratio) * num_samples)
        # test_sequences = sequences[train_size_abs:] # Niepotrzebne do grid search
        # test_targets = targets[train_size_abs:]

        if val_split_ratio <= 0 or val_split_ratio >= 1:
            raise ValueError("val_split must be > 0 and < 1 for grid search.")
        val_size_abs = int(val_split_ratio * train_size_abs)
        if val_size_abs == 0:
            val_size_abs = 1  # Ensure at least one validation sample
        train_sequences = sequences[: train_size_abs - val_size_abs]
        train_targets = targets[: train_size_abs - val_size_abs]
        val_sequences = sequences[train_size_abs - val_size_abs: train_size_abs]
        val_targets = targets[train_size_abs - val_size_abs: train_size_abs]

        if len(train_sequences) == 0 or len(val_sequences) == 0:
            raise ValueError("Train or Validation set is empty after splitting!")
        logging.info(
            f"Data prepared: Train size={len(train_sequences)}, Val size={len(val_sequences)}"
        )

    except Exception as e:
        logging.error(f"Fatal error during data preparation: {e}", exc_info=True)
        return

    grid_search_config = config.get("grid_search")
    if not grid_search_config or not isinstance(grid_search_config, dict):
        logging.error("`grid_search` section not found or not a dictionary in config.")
        return

    param_grid = {}
    for key, value in grid_search_config.items():
        if not isinstance(value, list):
            logging.warning(
                f"Value for '{key}' in grid_search is not a list. Converting to list: [{value}]"
            )
            param_grid[key] = [value]
        else:
            param_grid[key] = value

    if not param_grid:
        logging.error("Parameter grid for grid search is empty.")
        return

    logging.info(
        f"--- Starting Grid Search with {len(list(itertools.product(*param_grid.values())))} Combinations ---"
    )
    logging.info(f"Parameter Grid: {param_grid}")

    best_val_loss = float("inf")
    best_params_combination = None
    best_model_global_path = output_base_dir / f"{model_base_name}_best_grid_model.pth"
    results_summary = []

    # --- Grid Search Loop---
    start_time_grid = time.time()
    combination_counter = 0
    total_combinations = len(list(itertools.product(*param_grid.values())))

    for param_values in itertools.product(*param_grid.values()):
        combination_counter += 1
        current_params = dict(zip(param_grid.keys(), param_values))
        logging.info(
            f"\n--- Combination {combination_counter}/{total_combinations} ---"
        )
        logging.info(f"Testing parameters: {current_params}")
        start_time_comb = time.time()

        try:
            current_model_config = config["model"].copy()
            current_training_config = config["training"].copy()
            for key, value in current_params.items():
                if key in current_model_config:
                    current_model_config[key] = value
                if key in current_training_config:
                    current_training_config[key] = value

            current_batch_size = current_training_config["batch_size"]
            train_loader = DataLoader(
                MultiStockDataset(train_sequences, train_targets),
                batch_size=current_batch_size,
                shuffle=True,
                num_workers=current_training_config.get("num_workers", 0),
                pin_memory=device.type == "cuda",
                drop_last=True,
            )
            val_loader = DataLoader(
                MultiStockDataset(val_sequences, val_targets),
                batch_size=current_batch_size,
                shuffle=False,
                num_workers=current_training_config.get("num_workers", 0),
                pin_memory=device.type == "cuda",
            )

            common_build_params = {
                "stock_amount": stock_amount,
                "financial_features_amount": financial_features_amount,
                "lookback": lookback,
                "device": device,
            }
            model = build_model_dynamically(
                model_base_name, current_model_config, common_build_params
            )

            loss_type = current_training_config.get("loss_function", "RankLoss")
            if loss_type == "RankLoss":
                criterion = RankLoss(
                    lambda_rank=current_training_config.get("lambda_rank", 0.5)
                )
            elif loss_type == "MSE":
                criterion = torch.nn.MSELoss()
            else:
                raise ValueError(f"Unsupported loss function: {loss_type}")

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=current_training_config["learning_rate"],
                weight_decay=current_training_config.get("weight_decay", 0.01),
            )

            scheduler_config = current_training_config.get("lr_scheduler", {})
            scheduler_type = scheduler_config.get("type", None)
            scheduler = None
            if scheduler_type == "ReduceLROnPlateau":
                scheduler = ReduceLROnPlateau(
                    optimizer, **scheduler_config.get("params", {})
                )
            elif scheduler_type == "StepLR":
                scheduler = StepLR(optimizer, **scheduler_config.get("params", {}))
            elif scheduler_type == "ExponentialLR":
                scheduler = ExponentialLR(
                    optimizer, **scheduler_config.get("params", {})
                )

            if scheduler:
                logging.info(
                    f"Using LR scheduler: {scheduler_type} with params {scheduler_config.get('params', {})}"
                )
            else:
                logging.info("No LR scheduler used.")

            use_amp = (
                    current_training_config.get("use_amp", True) and device.type == "cuda"
            )
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            logging.info(f"Using Automatic Mixed Precision (AMP): {use_amp}")

            temp_save_dir = output_base_dir / f"temp_comb_{combination_counter}"
            temp_save_dir.mkdir(parents=True, exist_ok=True)

            _, current_best_val_loss = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                n_epochs=current_training_config["n_epochs"],
                save_path=temp_save_dir,
                patience=current_training_config["patience"],
                scaler=scaler,
                scheduler=scheduler,
                model_name=f"{model_base_name}_comb_{combination_counter}",
            )
            logging.info(
                f"Combination {combination_counter} finished. Best Val Loss: {current_best_val_loss:.6f}"
            )

            if current_best_val_loss < best_val_loss:
                best_val_loss = current_best_val_loss
                best_params_combination = current_params
                best_local_model_path = (
                        temp_save_dir
                        / f"{model_base_name}_comb_{combination_counter}_best_val.pth"
                )
                if best_local_model_path.exists():
                    torch.save(
                        torch.load(best_local_model_path, map_location="cpu"),
                        best_model_global_path,
                    )
                    logging.info(
                        f"   -> New overall best model saved with Val Loss: {best_val_loss:.6f}"
                    )
                else:
                    logging.warning(
                        f"   -> Best model file not found at {best_local_model_path}, cannot save global best."
                    )

            results_summary.append(
                {
                    "combination_id": combination_counter,
                    "params": current_params,
                    "best_val_loss": current_best_val_loss,
                }
            )

        except Exception as e:
            logging.error(
                f"Error during combination {combination_counter} ({current_params}): {e}",
                exc_info=True,
            )
            results_summary.append(
                {
                    "combination_id": combination_counter,
                    "params": current_params,
                    "best_val_loss": "ERROR",
                }
            )
        finally:
            if "model" in locals():
                del model
            if "optimizer" in locals():
                del optimizer
            if "criterion" in locals():
                del criterion
            if device.type == "cuda":
                torch.cuda.empty_cache()
            end_time_comb = time.time()
            logging.info(
                f"Time for combination {combination_counter}: {end_time_comb - start_time_comb:.2f} seconds."
            )

    end_time_grid = time.time()
    logging.info(f"\n--- Grid Search Finished ---")
    logging.info(f"Total time: {end_time_grid - start_time_grid:.2f} seconds.")
    if best_params_combination:
        logging.info(f"Best Validation Loss: {best_val_loss:.6f}")
        logging.info(f"Best Hyperparameters: {best_params_combination}")
        logging.info(f"Best model state saved to: {best_model_global_path}")

        best_params_path = output_base_dir / "best_grid_params.yaml"
        try:
            best_config = config.copy()
            best_config["grid_search_results"] = {
                "best_val_loss": float(best_val_loss),
                "best_params": best_params_combination,
            }
            for key, value in best_params_combination.items():
                if key in best_config["model"]:
                    best_config["model"][key] = value
                if key in best_config["training"]:
                    best_config["training"][key] = value

            with open(best_params_path, "w") as f:
                yaml.dump(best_config, f, default_flow_style=False)
            logging.info(f"Best configuration saved to: {best_params_path}")
        except Exception as e:
            logging.error(f"Error saving best parameters config: {e}")

    else:
        logging.warning(
            "Grid search completed, but no best parameters found (possibly all combinations failed)."
        )

    summary_path = output_base_dir / "grid_search_summary.csv"
    try:
        summary_df = pd.DataFrame(results_summary)
        params_df = summary_df["params"].apply(pd.Series)
        summary_df = pd.concat([summary_df.drop("params", axis=1), params_df], axis=1)
        summary_df = summary_df.sort_values(
            by="best_val_loss", ascending=True
        )
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Grid search summary saved to: {summary_path}")
    except Exception as e:
        logging.error(f"Error saving grid search summary: {e}")

    logging.info("--- Grid Search Script Finished ---")


# local run
# config="../experiments/configs/training_config_VanillaTransformer.yaml"
# run_grid_search(config)

# config = "../experiments/configs/training_config_TransformerCA.yaml"
# run_grid_search(config)

# config = "../experiments/configs/training_config_CrossFormer.yaml"
# run_grid_search(config)

# config = "../experiments/configs/training_config_MASTER.yaml"
# run_grid_search(config)

config = "../experiments/configs/training_config_iTransformer.yaml"
run_grid_search(config)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run Hyperparameter Grid Search for Port
#         folio Models"
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         required=True,
#         help="Path to the YAML configuration file containing the grid search setup.",
#     )
#     args = parser.parse_args()
#
#     run_grid_search(args.config)
#
#     # Przykład wywołania:
#     # python your_grid_search_script_name.py --config experiments/configs/grid_search_config_TransformerCA.yaml
