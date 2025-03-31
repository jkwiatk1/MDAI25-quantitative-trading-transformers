import logging
import os

import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
from types import SimpleNamespace

from experiments.utils.data_loading import (
    fill_missing_days,
    load_finance_data_xlsx,
    prepare_finance_data,
)
from experiments.utils.datasets import (
    prepare_sequential_data,
    prepare_combined_data,
    create_combined_sequences,
    normalize_data,
    MultiStockDataset,
)
from experiments.utils.feature_engineering import calc_input_features
from experiments.utils.metrics import RankLoss, WeightedMAELoss
from experiments.utils.training import (
    build_TransformerCA,
    train_model,
    evaluate_model,
    inverse_transform_predictions,
    plot_predictions,
)
from models.PortfolioCrossFormer import build_CrossFormer
from models.PortfolioMASTER import build_MASTER
from models.SimplePortfolioTransformer import build_SimplePortfolioTransformer


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        exit(1)


def get_tickers(config):
    if config["data"].get("yahoo_data", False):
        try:
            ticker_file = config["data"]["tickers"]
            tickers_df = pd.read_csv(ticker_file)
            if "Ticker" not in tickers_df.columns:
                logging.error(
                    f"Ticker file {ticker_file} must contain a 'Ticker' column."
                )
                exit(1)
            return tickers_df["Ticker"].tolist()
        except FileNotFoundError:
            logging.error(f"Ticker file not found at {ticker_file}")
            exit(1)
        except Exception as e:
            logging.error(f"Error reading ticker file {ticker_file}: {e}")
            exit(1)
    elif isinstance(config["data"]["tickers"], list):
        return config["data"]["tickers"]
    elif isinstance(config["data"]["tickers"], str):
        try:
            with open(config["data"]["tickers"], "r") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error(f"Ticker file not found at {config['data']['tickers']}")
            exit(1)
        except Exception as e:
            logging.error(f"Error reading ticker file {config['data']['tickers']}: {e}")
            exit(1)
    else:
        logging.error(
            "Invalid format for config['data']['tickers']. Expected list or file path."
        )
        exit(1)


def main(args):
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Setup paths
    output_dir = Path(config["data"]["output_dir"]) / config["model"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output path: {output_dir}")

    # Load data
    IS_DATA_FROM_YAHOO = config["data"]["yahoo_data"]
    logging.info("--- Starting Data Loading and Preparation ---")
    load_file = config["data"]["path"]
    start_date = pd.to_datetime(config["data"]["start_date"])
    end_date = pd.to_datetime(config["data"]["end_date"])
    logging.info(f"Loading data from: {load_file} between {start_date} and {end_date}")

    tickers_to_use = get_tickers(config)
    stock_amount = len(tickers_to_use)
    if stock_amount == 0:
        logging.error("No tickers specified or loaded.")
        return
    logging.info(f"Number of tickers: {stock_amount}")

    preproc_cols = config["data"].get("preproc_cols_to_use")
    if not preproc_cols or not isinstance(preproc_cols, list):
        logging.error("'preproc_cols_to_use' not defined or not a list in config.")
        return
    financial_features = len(preproc_cols)
    logging.info(f"Number of features per stock: {financial_features}")

    try:
        data_raw_dict, _ = load_finance_data_xlsx(
            load_file, config["data"].get("yahoo_data", False)
        )
        data_filled_dict = fill_missing_days(
            data_raw_dict, tickers_to_use, start_date, end_date
        )
        data = prepare_finance_data(
            data_filled_dict, tickers_to_use, config["data"]["init_cols_to_use"]
        )
        data = calc_input_features(
            df=data,
            tickers=tickers_to_use,
            cols=preproc_cols,
            time_step=config["training"]["lookback"],
        )
        data = {key: value[preproc_cols] for key, value in data.items()}

        data_scaled, feat_scalers = normalize_data(
            data, tickers_to_use, preproc_cols
        )

        # TODO to uwaga do iTransformera!
        # Przygotowanie sekwencji - ZAKŁADAMY, ŻE ZWRACA [samples, T, N, F] lub Dataset to obsłuży
        # Jeśli zwraca [samples, T, N*F], trzeba będzie dodać reshape w Dataset.__getitem__
        # Lub zmodyfikować prepare_sequential_data
        target_col = config["data"].get("preproc_target_col", "Daily profit")
        logging.info(
            f"Using '{target_col}' for target variable extraction (index 0 assumed)."
        )
        sequences, targets, ticker_mapping = prepare_sequential_data(
            data_scaled=data_scaled,
            tickers_to_use=tickers_to_use,
            lookback=config["training"]["lookback"],
            target_col_index=0,  # Zmień jeśli target jest inną kolumną w danych po `calc_input_features`
            # Dodaj argumenty, aby funkcja zwracała [samples, T, N, F] jeśli to możliwe
        )
        logging.info(
            f"Data sequences prepared. Shape: {sequences.shape}, Targets shape: {targets.shape}"
        )
        logging.info(
            f"Targets - Mean: {targets.mean():.6f}, Std: {targets.std():.6f}, Min: {targets.min():.6f}, Max: {targets.max():.6f}"
        )

    except Exception as e:
        logging.error(f"Error during data loading/preparation: {e}", exc_info=True)
        return

    # --- Data Splitting ---
    logging.info("--- Splitting Data ---")
    test_split_ratio = config["training"]["test_split"]
    val_split_ratio = config["training"]["val_split"]
    num_samples = len(sequences)

    if not (0 <= test_split_ratio < 1):
        logging.error("test_split must be between 0 and 1.")
        return
    if not (0 <= val_split_ratio < 1):
        logging.error("val_split must be between 0 and 1.")
        return

    train_size_abs = int((1 - test_split_ratio) * num_samples)
    test_sequences = sequences[train_size_abs:]
    test_targets = targets[train_size_abs:]

    if val_split_ratio > 0:
        val_size_abs = int(val_split_ratio * train_size_abs)
        if (
            val_size_abs == 0 and train_size_abs > 0
        ):  # Zapewnij co najmniej 1 próbkę walidacyjną jeśli to możliwe
            val_size_abs = 1
        train_sequences = sequences[: train_size_abs - val_size_abs]
        train_targets = targets[: train_size_abs - val_size_abs]
        val_sequences = sequences[train_size_abs - val_size_abs : train_size_abs]
        val_targets = targets[train_size_abs - val_size_abs : train_size_abs]
        logging.info(
            f"Train size: {len(train_sequences)}, Validation size: {len(val_sequences)}, Test size: {len(test_sequences)}"
        )
        if len(train_sequences) == 0 or len(val_sequences) == 0:
            logging.warning("Train or Validation set is empty after splitting!")
    else:
        train_sequences = sequences[:train_size_abs]
        train_targets = targets[:train_size_abs]
        val_sequences, val_targets = None, None  # Brak walidacji
        logging.info(
            f"Train size: {len(train_sequences)}, Test size: {len(test_sequences)} (No validation set)"
        )
        if len(train_sequences) == 0:
            logging.warning("Train set is empty after splitting!")

    if len(test_sequences) == 0:
        logging.warning("Test set is empty after splitting!")

    # --- Create DataLoaders ---
    logging.info("--- Creating DataLoaders ---")
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get(
        "num_workers", 0
    )  # Dodaj do configu, domyślnie 0
    pin_memory = torch.cuda.is_available()  # Użyj pin_memory tylko z GPU

    try:
        train_loader = DataLoader(
            MultiStockDataset(train_sequences, train_targets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # drop_last często pomaga
        )
        val_loader = None
        if val_sequences is not None and len(val_sequences) > 0:
            val_loader = DataLoader(
                MultiStockDataset(val_sequences, val_targets),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        test_loader = None
        if test_sequences is not None and len(test_sequences) > 0:
            test_loader = DataLoader(
                MultiStockDataset(test_sequences, test_targets),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
    except Exception as e:
        logging.error(f"Error creating DataLoaders: {e}", exc_info=True)
        return

    # --- Log Parameters ---
    logging.info("--- Logging Parameters ---")
    logging.info(f"Model Config: {config['model']}")
    logging.info(f"Training Config: {config['training']}")

    # --- Build Model (Dynamic) ---
    logging.info("--- Building Model ---")
    model_type = config["model"].get("name", "TransformerCA")
    try:
        if model_type == "TransformerCA":
            logging.info("Building Portfolio TransformerCA...")
            model = build_TransformerCA(
                stock_amount=stock_amount,
                financial_features_amount=financial_features,
                lookback=config["training"]["lookback"],
                d_model=config["model"]["d_model"],
                n_heads=config["model"]["n_heads"],
                d_ff=config["model"]["d_ff"],
                dropout=config["model"]["dropout"],
                num_encoder_layers=config["model"]["num_encoder_layers"],
                device=device,
            )
        elif model_type == "CrossFormer":
            logging.info("Building Portfolio Crossformer...")
            if not all(k in config["model"] for k in ("seg_len", "win_size", "factor")):
                logging.error(
                    "Missing required parameters (seg_len, win_size, factor) for PortfolioCrossformer."
                )
                return
            model = build_CrossFormer(
                stock_amount=stock_amount,
                financial_features=financial_features,
                in_len=config["training"]["lookback"],
                seg_len=config["model"]["seg_len"],
                win_size=config["model"]["win_size"],
                factor=config["model"]["factor"],
                aggregation_type='avg_pool',  # TODO add to config
                d_model=config["model"]["d_model"],
                d_ff=config["model"]["d_ff"],
                n_heads=config["model"]["n_heads"],
                # Użyj num_encoder_layers jako e_layers dla spójności
                e_layers=config["model"]["num_encoder_layers"],
                dropout=config["model"]["dropout"],
                device=device,
            )
        elif model_type == "MASTER":
            logging.info("Building Portfolio MASTER...")
            model = build_MASTER(
                stock_amount=stock_amount,
                financial_features_amount=financial_features,
                lookback=config["training"]["lookback"],
                d_model=config["model"]["d_model"],
                n_heads=config["model"].get("n_heads", config["model"]["n_heads"]),
                dropout=config["model"].get("dropout", config["model"]["dropout"]),
                d_ff=config["model"]["d_ff"],
                num_encoder_layers=config['model']['num_encoder_layers'],
                device=device,
            )
        else:
            logging.error(f"Unknown model name: {model_type}")
            return

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Built {model_type} with {total_params:,} trainable parameters.")

    except KeyError as e:
        logging.error(
            f"Missing key in model configuration: {e}. Please check config.yaml."
        )
        return
    except Exception as e:
        logging.error(f"Error building model {model_type}: {e}", exc_info=True)
        return

    # --- Training Setup ---
    logging.info("--- Setting up Training ---")
    try:
        loss_type = config["training"].get(
            "loss_function", "RankLoss"
        )
        if loss_type == "RankLoss":
            criterion = RankLoss(lambda_rank=config["training"].get("lambda_rank", 0.5))
        elif loss_type == "MSE":
            criterion = torch.nn.MSELoss()
        # Dodaj inne opcje np. 'MAE'
        # elif loss_type == 'MAE':
        #    criterion = torch.nn.L1Loss()
        else:
            logging.error(f"Unsupported loss function: {loss_type}")
            return
        logging.info(f"Using loss function: {criterion.__class__.__name__}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get(
                "weight_decay", 0.01
            ),
        )
        logging.info(
            f"Using optimizer: AdamW with lr={config['training']['learning_rate']} and weight_decay={config['training'].get('weight_decay', 0.01)}"
        )

        # Konfiguracja schedulera
        scheduler_config = config["training"].get("lr_scheduler", {})
        scheduler_type = scheduler_config.get("type", None)
        scheduler = None
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer, **scheduler_config.get("params", {})
            )
        elif scheduler_type == "StepLR":
            scheduler = StepLR(optimizer, **scheduler_config.get("params", {}))
        elif scheduler_type == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, **scheduler_config.get("params", {}))

        if scheduler:
            logging.info(
                f"Using LR scheduler: {scheduler_type} with params {scheduler_config.get('params', {})}"
            )
        else:
            logging.info("No LR scheduler used.")

        use_amp = config["training"].get("use_amp", True) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        logging.info(f"Using Automatic Mixed Precision (AMP): {use_amp}")

    except Exception as e:
        logging.error(f"Error setting up training components: {e}", exc_info=True)
        return

    # --- Training Loop ---
    logging.info("--- Starting Training ---")
    best_model_path = None
    try:
        best_model_path, history = train_model(  # Zwraca też historię
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,  # Może być None
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_epochs=config["training"]["n_epochs"],
            save_path=output_dir,
            patience=config["training"]["patience"],
            scaler=scaler,
            scheduler=scheduler,
            model_name=model_type,  # Przekaż nazwę modelu do zapisania
        )
        logging.info(f"Training finished. Best model saved to: {best_model_path}")
        # Możesz zapisać historię treningu/walidacji (history)
        # np.save(output_dir / f"{model_type}_training_history.npy", history)
    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        # Możesz chcieć kontynuować do ewaluacji jeśli jakiś model został zapisany
        # Sprawdź, czy jakikolwiek model .pth istnieje w output_dir

    # --- Evaluation ---
    logging.info("--- Starting Evaluation ---")
    if best_model_path and Path(best_model_path).exists() and test_loader:
        logging.info(f"Loading best model from: {best_model_path}")
        try:
            # Załaduj stan na CPU, a następnie przenieś na właściwe urządzenie
            # To bezpieczniejsze, jeśli model był trenowany na GPU, a ewaluacja jest na CPU
            model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
            model.to(device)  # Upewnij się, że model jest na właściwym urządzeniu

            test_predictions, test_targets_eval, test_loss = evaluate_model(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
                scaler=scaler,  # scaler jest używany tylko do autocast w evaluate_model
            )
            logging.info(f"Evaluation finished. Test Loss: {test_loss:.6f}")

            # --- Post-processing and Plotting ---
            logging.info("--- Post-processing and Plotting Results ---")

            # Squeeze jeśli model zwraca [B, N, 1], a targets mają [B, N]
            # Sprawdź kształty przed squeeze!
            logging.info(
                f"Raw predictions shape: {test_predictions.shape}"
            )  # Powinno być (num_test_samples, N, 1)
            logging.info(
                f"Raw targets shape: {test_targets_eval.shape}"
            )  # Powinno być (num_test_samples, N)
            if test_predictions.shape[-1] == 1:
                test_predictions = test_predictions.squeeze(
                    -1
                )  # -> (num_test_samples, N, 1)
                test_targets_eval = test_targets_eval.squeeze(-1)

            # Odwróć transformację tylko dla kolumny docelowej
            target_col_name = config["data"].get("preproc_target_col")
            if not target_col_name:
                logging.warning(
                    "'preproc_target_col' not specified in config.  Skipping inverse transform and using scaled values for plotting."
                )
                test_predictions_to_plot = test_predictions
                test_targets_to_plot = test_targets_eval

            else:
                try:
                    logging.info(
                        f"Attempting inverse transform for target '{target_col_name}'..."
                    )
                    # test_predictions = test_predictions.squeeze(-1)
                    #

                    (
                        test_predictions_inv,
                        test_targets_inv,
                    ) = inverse_transform_predictions(
                        test_predictions,
                        test_targets_eval,
                        tickers_to_use,
                        feat_scalers,
                        target_col_name,
                    )
                    test_predictions_to_plot = test_predictions_inv
                    test_targets_to_plot = test_targets_inv
                    logging.info("Inverse transform finished.")

                except Exception as e:
                    logging.error(
                        f"Unexpected error calling inverse_transform_predictions: {e}",
                        exc_info=True,
                    )
                    test_predictions_to_plot = test_predictions
                    test_targets_to_plot = test_targets_eval

                # --- Plotting
                try:
                    first_ticker = tickers_to_use[0]
                    if (
                        first_ticker in data_scaled
                        and len(data_scaled[first_ticker]) >= num_samples
                    ):
                        test_dates = data_scaled[first_ticker].index[train_size_abs:]
                        if len(test_dates) == len(test_predictions_to_plot):
                            plot_predictions(
                                test_predictions_to_plot,
                                test_targets_to_plot,
                                tickers_to_use,
                                save_path=output_dir,
                                dates=test_dates,
                                model_name=model_type,
                            )
                        else:
                            logging.warning(
                                "Length mismatch between test dates and predictions after potential drops."
                            )
                            plot_predictions(
                                test_predictions_to_plot,
                                test_targets_to_plot,
                                tickers_to_use,
                                save_path=output_dir,
                                model_name=model_type,
                            )
                    else:
                        logging.warning("Could not extract test dates for plotting.")
                        plot_predictions(
                            test_predictions_to_plot,
                            test_targets_to_plot,
                            tickers_to_use,
                            save_path=output_dir,
                            model_name=model_type,
                        )

                except Exception as e:
                    logging.error(f"Error during plotting: {e}", exc_info=True)

        except FileNotFoundError:
            logging.error(
                f"Best model file not found at {best_model_path}. Skipping evaluation."
            )
        except Exception as e:
            logging.error(
                f"Error during model loading or evaluation: {e}", exc_info=True
            )
    elif not test_loader:
        logging.warning("Test loader is empty. Skipping evaluation.")
    else:
        logging.warning(
            "Best model path not found or training failed. Skipping evaluation."
        )

    logging.info("--- Pipeline Finished ---")


# local run
# model_name = "TransformerCA"
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# log_file = os.path.join(
#     base_dir, "data", "exp_result", "test", model_name, "logs", "pipeline.log"
# )
# setup_logging(log_file)
# args = SimpleNamespace(config="../experiments/configs/training_config_TransformerCA.yaml")
# main(args)

# model_name = "CrossFormer"
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# log_file = os.path.join(
#     base_dir, "data", "exp_result", "test", model_name, "logs", "pipeline.log"
# )
# setup_logging(log_file)
# args = SimpleNamespace(config="../experiments/configs/training_config_CrossFormer.yaml")
# main(args)

model_name = "MASTER"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(
    base_dir, "data", "exp_result", "test", model_name, "logs", "pipeline.log"
)
setup_logging(log_file)
args = SimpleNamespace(config="../experiments/configs/training_config_MASTER.yaml")
main(args)

"""
# --- Uruchomienie Skryptu ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Portfolio Model Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., experiments/configs/my_config.yaml)",
    )
    args = parser.parse_args()

    # Załaduj config na początku
    pipeline_config = load_config(args.config)

    if pipeline_config:
        # Ustaw logowanie na podstawie ścieżki z configu
        model_name_from_config = pipeline_config.get("model", {}).get("name", "DefaultModel")
        log_dir = Path(pipeline_config.get("data", {}).get("output_dir", "./output")) / "logs"
        log_file = log_dir / f"{model_name_from_config}_pipeline.log"
        setup_logging(log_file)

        # Uruchom główną funkcję
        main(pipeline_config)
    else:
        print("Failed to load configuration. Exiting.")
        exit(1)
"""

# if __name__ == "__main__":
#     model_name = "Transformer"
#     parser = argparse.ArgumentParser(description="Run QuantFormer Training Pipeline")
#     parser.add_argument(
#         "--config",
#         type=str,
#         help="Path to the YAML configuration file",
#     )
#     args = parser.parse_args()
#
#     setup_logging(f"./data/exp_result/{model_name}/logs/pipeline.log")
#     main(args)
