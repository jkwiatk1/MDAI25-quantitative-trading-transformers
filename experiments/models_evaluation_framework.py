import logging
import os
from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

# --- Importy funkcji pomocniczych (załóżmy, że są w odpowiednich ścieżkach) ---
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

# --- Importy funkcji budujących modele ---
from models.PortfolioVanillaTransformer import build_PortfolioVanillaTransformer
from models.PortfolioCrossFormer import build_CrossFormer # Zmieniono nazwę pliku dla spójności
from models.PortfolioMASTER import build_MASTER # Zmieniono nazwę pliku dla spójności
from models.PortfolioiTransformer import build_PortfolioITransformer # Zmieniono nazwę pliku dla spójności
from models.PortfolioTransformerCA import build_TransformerCA # Zmieniono nazwę pliku dla spójności


def setup_logging(log_file):
    """Konfiguruje logowanie do pliku i konsoli."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    # Usuń poprzednie handlery, jeśli istnieją (przydatne przy wielokrotnym wywoływaniu)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()], # mode='w' nadpisuje log przy każdym uruchomieniu
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
    """Główna funkcja przeprowadzająca ewaluację modelu."""
    try:
        config = load_config(config_path)
    except Exception:
        return # Zakończ, jeśli nie udało się wczytać configu

    # --- Setup ---
    model_name = config["model"]["name"]
    output_base_dir = Path(config["data"].get("output_dir", "results")) # Użyj domyślnej nazwy, jeśli brak
    output_dir = output_base_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Konfiguracja logowania specyficzna dla uruchomienia
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

    # --- Wczytywanie Tickerów ---
    try:
        selected_tickers = get_tickers(config) # Użyj funkcji pomocniczej, jeśli istnieje
        stock_amount = len(selected_tickers)
        if stock_amount == 0:
            logging.error("No tickers specified or loaded.")
            return
        logging.info(f"Processing {stock_amount} tickers.")
    except Exception as e:
        logging.error(f"Error getting tickers: {e}", exc_info=True)
        return

    # --- Ładowanie i Przygotowanie Danych ---
    # Ten blok jest kosztowny, idealnie dane testowe i scalery powinny być zapisane
    # podczas treningu i tutaj tylko wczytane. Na razie zostawiamy jak jest.
    logging.info("--- Loading and Preparing Data ---")
    try:
        data_raw_dict, _ = load_finance_data_xlsx(
            config["data"]["path"], config["data"].get("yahoo_data", False)
        )
        data_filled_dict = fill_missing_days(data_raw_dict, selected_tickers, start_date, end_date)
        data = prepare_finance_data(data_filled_dict, selected_tickers, config["data"]["init_cols_to_use"])

        # Upewnij się, że `preproc_cols_to_use` jest listą
        preproc_cols = config["data"].get("preproc_cols_to_use", [])
        if not isinstance(preproc_cols, list):
             logging.error("`preproc_cols_to_use` must be a list in config.")
             return
        financial_features_amount = len(preproc_cols)
        if financial_features_amount == 0:
            logging.error("`preproc_cols_to_use` cannot be empty.")
            return

        data = calc_input_features(data, selected_tickers, preproc_cols, lookback)
        data = {key: value[preproc_cols] for key, value in data.items() if key in data} # Bezpieczniejsze pobieranie

        # --- Normalizacja i Sekwencjonowanie ---
        data_scaled, feat_scalers = normalize_data(data, selected_tickers, preproc_cols)
        target_col_name = config["data"].get("preproc_target_col", preproc_cols[0]) # Domyślnie pierwsza kolumna
        if target_col_name not in preproc_cols:
            logging.error(f"Target column '{target_col_name}' not found in preprocessed columns: {preproc_cols}")
            return
        target_col_index = preproc_cols.index(target_col_name)
        logging.info(f"Using '{target_col_name}' (index {target_col_index}) as target variable.")

        sequences, targets, _ = prepare_sequential_data(
            data_scaled, selected_tickers, lookback, target_col_index
        )
        logging.info(f"Data sequences prepared. Shape: {sequences.shape}, Targets shape: {targets.shape}")

        # --- Podział na Zbiór Testowy ---
        # Zakładamy, że podział jest taki sam jak podczas treningu
        test_split_ratio = config["training"]["test_split"]
        num_samples = len(sequences)
        train_size_abs = int((1 - test_split_ratio) * num_samples)
        test_sequences = sequences[train_size_abs:]
        test_targets = targets[train_size_abs:]

        if len(test_sequences) == 0:
            logging.error("Test set is empty after splitting. Check data length and split ratios.")
            return
        logging.info(f"Test set size: {len(test_sequences)}")

        test_loader = DataLoader(
            MultiStockDataset(test_sequences, test_targets),
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"].get("num_workers", 0),
            pin_memory=device.type == 'cuda' # Pin memory tylko dla GPU
        )

        # Przygotuj zakres dat dla wykresów (jeśli możliwe)
        first_ticker = selected_tickers[0]
        all_dates = data[first_ticker].index if first_ticker in data else None
        test_dates = all_dates[train_size_abs:] if all_dates is not None else None
        if test_dates is not None and len(test_dates) != len(test_sequences):
             logging.warning(f"Length mismatch between expected test dates ({len(test_dates)}) and actual test sequences ({len(test_sequences)}). Plotting without dates.")
             test_dates = None

    except Exception as e:
        logging.error(f"Error during data loading/preparation: {e}", exc_info=True)
        return

    # --- Dynamiczne Budowanie Modelu ---
    logging.info(f"--- Building Model: {model_name} ---")
    model_params = config["model"]
    try:
        # Przygotuj wspólne parametry
        common_params_build = {
            "stock_amount": stock_amount,
            "financial_features_amount": financial_features_amount,
            "lookback": lookback,
            "device": device,
        }
        # Połącz wspólne i specyficzne parametry, usuń 'name'
        builder_args = {**common_params_build, **model_params}
        builder_args.pop('name', None) # Usuń 'name', bo nie jest argumentem buildera

        # Wywołaj odpowiednią funkcję build
        if model_name == "VanillaTransformer":
            build_PortfolioVanillaTransformer(
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
            # Dostosuj nazwy parametrów, jeśli są inne w build_CrossFormer
            model = build_CrossFormer(
                stock_amount=stock_amount,
                financial_features=len(config["data"]["preproc_cols_to_use"]),
                in_len=config["training"]["lookback"],
                seg_len=config["model"]["seg_len"],
                win_size=config["model"]["win_size"],
                factor=config["model"]["factor"],
                aggregation_type="avg_pool",  # TODO add to config
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

    # --- Ładowanie Wag ---
    weights_path = config["model"].get("weights_path")
    if not weights_path or not Path(weights_path).exists():
        logging.error(f"Model weights path '{weights_path}' not found or not specified in config.")
        return
    try:
        # Załaduj stan na CPU, a następnie przenieś na właściwe urządzenie
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.to(device)
        logging.info(f"Successfully loaded model weights from {weights_path}")
    except Exception as e:
        logging.error(f"Error loading model weights from {weights_path}: {e}", exc_info=True)
        return

    # --- Ewaluacja Modelu ---
    logging.info("--- Evaluating Model ---")
    try:
        # Użyj tej samej funkcji straty co podczas treningu (lub MSE do prostej ewaluacji)
        eval_loss_type = config["training"].get("loss_function", "RankLoss")
        if eval_loss_type == "RankLoss":
            criterion = RankLoss(lambda_rank=config["training"].get("lambda_rank", 0.5))
        elif eval_loss_type == "MSE":
            criterion = torch.nn.MSELoss()
        else:
            logging.warning(f"Using MSE loss for evaluation as '{eval_loss_type}' is not MSE or RankLoss.")
            criterion = torch.nn.MSELoss()

        use_amp_eval = config["training"].get("use_amp", True) and device.type == 'cuda'
        # Scaler jest potrzebny tylko do autocast w evaluate_model
        scaler_eval = torch.cuda.amp.GradScaler(enabled=use_amp_eval)

        predictions_scaled, targets_scaled, test_loss = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            scaler=scaler_eval # Przekaż scaler
        )
        logging.info(f"Evaluation finished. Test Loss ({criterion.__class__.__name__}): {test_loss:.6f}")

        # Squeeze ostatniego wymiaru, jeśli jest 1
        if predictions_scaled.shape[-1] == 1:
            predictions_scaled = predictions_scaled.squeeze(-1)
        if targets_scaled.shape[-1] == 1:
            targets_scaled = targets_scaled.squeeze(-1)

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}", exc_info=True)
        return

    # --- Obliczanie Metryk ---
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
        portfolio_top_k = config["portfolio"].get("top_k", 5) # Pobierz k z sekcji portfolio configu
        precision_at_k_value = calculate_precision_at_k(predictions_scaled, targets_inv, top_k=portfolio_top_k)
        predictive_metrics[f'Precision@{portfolio_top_k}'] = precision_at_k_value

        # Metryki portfelowe (używają oryginalnych predykcji i odwróconych targetów)
        portfolio_risk_free_rate = config["portfolio"].get("risk_free_rate", 0.0) # Pobierz Rf z sekcji portfolio
        portfolio_metrics, portfolio_value_curve = calculate_portfolio_performance(
            predictions_scaled, targets_inv, top_k=portfolio_top_k, risk_free_rate=portfolio_risk_free_rate
        )

        # Połącz wszystkie metryki
        all_metrics = {**portfolio_metrics, **predictive_metrics, f"Test Loss ({eval_loss_type})": test_loss}
        logging.info(f"Final Combined Metrics: {all_metrics}")

    except Exception as e:
        logging.error(f"Error calculating metrics: {e}", exc_info=True)
        return

    # --- Zapisywanie Wyników ---
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

        # Zapisz też w formacie CSV dla łatwiejszej agregacji
        csv_results_path = output_dir / "evaluation_results.csv"
        pd.DataFrame([all_metrics]).to_csv(csv_results_path, index=False)
        logging.info(f"Saved evaluation metrics to CSV: {csv_results_path}")

        # Zapisz krzywą wartości portfela
        curve_df = pd.DataFrame({'PortfolioValue': portfolio_value_curve})
        if test_dates is not None and len(test_dates) == len(portfolio_value_curve) -1 : # Curve ma +1 element (start)
             curve_df.index = pd.to_datetime(['start'] + list(test_dates)) # Dodaj placeholder dla startu
        curve_path = output_dir / f"{model_name}_portfolio_value_curve.csv"
        curve_df.to_csv(curve_path)
        logging.info(f"Saved portfolio value curve data to {curve_path}")

    except Exception as e:
        logging.error(f"Error saving results: {e}", exc_info=True)

    # --- Wizualizacja ---
    try:
        plt.figure(figsize=(12, 6))
        if test_dates is not None and len(test_dates) == len(portfolio_value_curve) - 1:
             plot_dates = pd.to_datetime(list(test_dates)) # Użyj tylko dat testowych
             plt.plot(plot_dates, portfolio_value_curve[1:], linestyle="-", label=f"{model_name} Portfolio Value") # Pomiń punkt startowy dla osi X
             plt.xlabel("Date")
             plt.xticks(rotation=45)
        else:
             plt.plot(portfolio_value_curve[1:], linestyle="-", label=f"{model_name} Portfolio Value") # Pomiń punkt startowy dla osi X
             plt.xlabel(f"Trading Days (Test Period, {len(portfolio_value_curve)-1} days)")

        plt.ylabel("Portfolio Value (Starts at 1.0)")
        plt.title(f"Portfolio Value Over Time ({model_name}, Top-{portfolio_top_k} Strategy)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = output_dir / f"{model_name}_portfolio_value_curve.png"
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Saved portfolio value plot to {plot_path}")
        # plt.show() # Odkomentuj, jeśli chcesz pokazywać wykresy interaktywnie
        plt.close() # Zamknij figurę po zapisaniu

    except Exception as e:
        logging.error(f"Error during plotting: {e}", exc_info=True)

    logging.info(f"--- Evaluation script finished for {model_name} ---")


# # local run
# model_name = "VanillaTransformer"
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# log_file = os.path.join(
#     base_dir, "data", "exp_result", "test", model_name, "logs", "evaluation.log"
# )
# setup_logging(log_file)
# args = SimpleNamespace(config="../experiments/configs/test_config_VanillaTransformer.yaml")
# main(args)

# # local run
# model_name = "TransformerCA"
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# log_file = os.path.join(
#     base_dir, "data", "exp_result", "test", model_name, "logs", "evaluation.log"
# )
# setup_logging(log_file)
# args = SimpleNamespace(config="../experiments/configs/test_config_TransformerCA.yaml")
# main(config_path="../experiments/configs/test_config_TransformerCA.yaml")

if __name__ == "__main__": # Dobrą praktyką jest nadal umieszczać kod wykonawczy w tym bloku

    # === Konfiguracja dla
    # model_name_to_run = "VanillaTransformer"  # Używaj spójnych nazw z config['model']['name']
    # config_file_path = "../experiments/configs/test_config_VanillaTransformer.yaml"  # Ścieżka do pliku konfiguracyjnego

    # Modelu 1: TransformerCA ===
    # model_name_to_run = "TransformerCA" # Używaj spójnych nazw z config['model']['name']
    # config_file_path = "../experiments/configs/test_config_TransformerCA.yaml" # Ścieżka do pliku konfiguracyjnego

    # model_name_to_run = "iTransformer"  # Używaj spójnych nazw z config['model']['name']
    # config_file_path = "../experiments/configs/test_config_iTransformer.yaml"  # Ścieżka do pliku konfiguracyjnego

    # model_name_to_run = "CrossFormer"  # Używaj spójnych nazw z config['model']['name']
    # config_file_path = "../experiments/configs/test_config_CrossFormer.yaml"  # Ścieżka do pliku konfiguracyjnego
    #
    model_name_to_run = "MASTER"  # Używaj spójnych nazw z config['model']['name']
    config_file_path = "../experiments/configs/test_config_MASTER.yaml"  # Ścieżka do pliku konfiguracyjnego
    #

    # -- Konfiguracja logowania (opcjonalna, main() zrobi to ponownie) --
    # base_dir = Path(__file__).resolve().parent.parent # Bardziej niezawodne określenie base_dir
    # log_dir_local = base_dir / "data" / "exp_result" / "test_local" / model_name_to_run / "logs"
    # log_file_local = log_dir_local / "evaluation_local.log"
    # setup_logging(log_file_local) # Można skonfigurować logowanie tutaj LUB polegać na tym w main()
    # print(f"Local run configured for {model_name_to_run}. Log file (tentative): {log_file_local}")

    print(f"\n--- Running Evaluation for: {model_name_to_run} ---")
    print(f"--- Using Config: {config_file_path} ---\n")

    # Sprawdź, czy plik konfiguracyjny istnieje
    if not Path(config_file_path).exists():
        print(f"ERROR: Configuration file not found at {config_file_path}")
    else:
        # Wywołaj funkcję main, przekazując ścieżkę do configu
        try:
             main(config_path=config_file_path)
        except Exception as e:
             # Logowanie błędu krytycznego, jeśli main() rzuci wyjątek, którego nie złapał wewnętrznie
             logging.error(f"Critical error during main execution for {model_name_to_run}: {e}", exc_info=True)
             print(f"CRITICAL ERROR running {model_name_to_run}. Check logs for details.")

    print(f"\n--- Finished Evaluation for: {model_name_to_run} ---")

    # === Możesz dodać bloki dla innych modeli ===
    # model_name_to_run = "PortfolioVanillaTransformer"
    # config_file_path = "../experiments/configs/test_config_VanillaTransformer.yaml"
    # print(f"\n--- Running Evaluation for: {model_name_to_run} ---")
    # print(f"--- Using Config: {config_file_path} ---\n")
    # if not Path(config_file_path).exists():
    #     print(f"ERROR: Configuration file not found at {config_file_path}")
    # else:
    #     try:
    #         main(config_path=config_file_path)
    #     except Exception as e:
    #         logging.error(f"Critical error during main execution for {model_name_to_run}: {e}", exc_info=True)
    #         print(f"CRITICAL ERROR running {model_name_to_run}. Check logs for details.")
    # print(f"\n--- Finished Evaluation for: {model_name_to_run} ---")

    # ... i tak dalej dla pozostałych modeli ...

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