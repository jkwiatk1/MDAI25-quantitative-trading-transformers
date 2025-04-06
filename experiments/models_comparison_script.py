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
    """Konfiguruje logowanie do pliku i konsoli."""
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
    """Wczytuje konfigurację z pliku YAML."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded comparison configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading comparison config {config_path}: {e}", exc_info=True)
        raise

def load_portfolio_curve(file_path):
    """Wczytuje krzywą wartości portfela z CSV."""
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True) # Zakładamy, że pierwsza kolumna to data/indeks
        # Upewnij się, że nazywa się 'PortfolioValue' lub dostosuj
        if 'PortfolioValue' not in df.columns:
             # Spróbuj znaleźć pierwszą kolumnę numeryczną, jeśli nazwa jest inna
             num_cols = df.select_dtypes(include=np.number).columns
             if not num_cols.empty:
                  col_name = num_cols[0]
                  logging.warning(f"Column 'PortfolioValue' not found in {file_path}. Using first numeric column: '{col_name}'.")
                  df.rename(columns={col_name: 'PortfolioValue'}, inplace=True)
             else:
                  raise ValueError(f"No numeric portfolio value column found in {file_path}")

        # Usuń placeholder 'start' jeśli istnieje
        if 'start' in df.index:
             df = df.drop('start')
        df.index = pd.to_datetime(df.index)
        return df['PortfolioValue']
    except Exception as e:
        logging.error(f"Error loading portfolio curve from {file_path}: {e}")
        return None

def load_benchmark_data(file_path, index_col_name='S&P500', date_col='Date'):
    """Wczytuje i przetwarza dane benchmarku."""
    try:
        df = pd.read_excel(file_path)

        # Konwersja daty
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        # Konwersja wartości indeksu (obsługa przecinków jako separatorów tysięcy)
        if df[index_col_name].dtype == 'object':
            df[index_col_name] = df[index_col_name].str.replace(',', '', regex=False).astype(float)
        else:
            df[index_col_name] = df[index_col_name].astype(float)

        # Sortowanie wg daty
        df = df.sort_index()

        # Obliczenie dziennych zwrotów
        df['DailyReturn'] = df[index_col_name].pct_change().fillna(0)

        # Obliczenie skumulowanej wartości (zaczynając od 1.0)
        df['ValueCurve'] = (1 + df['DailyReturn']).cumprod()
        # Upewnij się, że zaczyna się od 1.0 - znajdź pierwszy niezerowy zwrot
        first_valid_index = df['DailyReturn'].ne(0).idxmax()
        # Przesuń krzywą, aby zaczynała się od 1 w dniu poprzedzającym pierwszy zwrot
        start_value = df.loc[:first_valid_index, 'ValueCurve'].iloc[-2] # wartość przed pierwszym zwrotem
        df['ValueCurve'] = df['ValueCurve'] / start_value

        # Jeśli pierwszy dzień ma być 1.0
        df.loc[df.index < first_valid_index, 'ValueCurve'] = 1.0

        return df[['ValueCurve']] # Zwróć tylko krzywą wartości z poprawnym indeksem daty
    except Exception as e:
        logging.error(f"Error loading or processing benchmark data from {file_path}: {e}", exc_info=True)
        return None

def main_comparison(config):
    """Główna funkcja skryptu porównawczego."""
    results_base_dir = Path(config['results_base_dir'])
    model_names = config['models_to_compare']
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

    # --- Wczytaj Dane Benchmarku ---
    logging.info(f"Loading benchmark data from: {benchmark_file}")
    df_benchmark_curve = load_benchmark_data(benchmark_file, benchmark_col, benchmark_date_col)
    if df_benchmark_curve is None:
        logging.error("Failed to load benchmark data. Exiting.")
        return

    # --- Wczytaj Wyniki Modeli ---
    model_curves = {}
    min_date = pd.Timestamp.max
    max_date = pd.Timestamp.min

    for model_name in model_names:
        curve_path = results_base_dir / model_name / f"{model_name}_portfolio_value_curve.csv"
        if curve_path.exists():
            logging.info(f"Loading results for {model_name} from {curve_path}")
            curve = load_portfolio_curve(curve_path)
            if curve is not None:
                model_curves[model_name] = curve
                # Aktualizuj zakres dat na podstawie wczytanych krzywych modeli
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

    # --- Przytnij Benchmark do Wspólnego Okresu ---
    # Dodaj jeden dzień przed min_date, aby uzyskać wartość początkową 1.0 dla benchmarku
    start_date_bm = df_benchmark_curve.index[df_benchmark_curve.index < min_date].max()
    if pd.isna(start_date_bm):
        start_date_bm = min_date # Jeśli nie ma wcześniejszej daty
    df_benchmark_curve_aligned = df_benchmark_curve.loc[start_date_bm:max_date].copy()
    # Znormalizuj benchmark, aby zaczynał się od 1.0 na początku okresu modeli
    df_benchmark_curve_aligned['ValueCurve'] = df_benchmark_curve_aligned['ValueCurve'] / df_benchmark_curve_aligned['ValueCurve'].iloc[0]
    # Usuń datę przed startem modeli, jeśli została dodana
    df_benchmark_curve_aligned = df_benchmark_curve_aligned[df_benchmark_curve_aligned.index >= min_date]


    # --- Generuj Wykres Porównawczy ---
    plt.style.use('seaborn-v0_8-darkgrid') # Użyj stylu dla lepszego wyglądu
    plt.figure(figsize=(14, 8))

    # Wykres Benchmarku
    plt.plot(df_benchmark_curve_aligned.index, df_benchmark_curve_aligned['ValueCurve'],
             label=f"{benchmark_col} (Buy & Hold)", linewidth=2, linestyle='--', color='black')

    # Wykresy Modeli
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_curves))) # Paleta kolorów
    for i, (model_name, curve) in enumerate(model_curves.items()):
        # Przytnij krzywą modelu do wspólnego okresu
        curve_aligned = curve.loc[min_date:max_date]
        # Znormalizuj, aby zaczynała się od 1.0 na początku okresu
        curve_aligned = curve_aligned / curve_aligned.iloc[0]
        plt.plot(curve_aligned.index, curve_aligned, label=model_name, linewidth=1.5, color=colors[i])

    # Ustawienia Wykresu
    plt.title(f"Portfolio Value Comparison ({min_date.date()} to {max_date.date()})", fontsize=16)
    plt.ylabel("Portfolio Value (Normalized to 1.0 at Start)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.yscale('log') # Skala logarytmiczna często jest lepsza dla zwrotów skumulowanych
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()

    # Formatowanie osi X dla dat
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(df_benchmark_curve_aligned)//12//6))) # Około 6-12 etykiet
    plt.gcf().autofmt_xdate() # Automatyczne formatowanie etykiet dat

    # Zapisz wykres
    comparison_plot_path = output_dir_comp / "model_vs_benchmark_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300)
    logging.info(f"Comparison plot saved to {comparison_plot_path}")
    # plt.show() # Odkomentuj, jeśli chcesz wyświetlić
    plt.close()

    # --- (Opcjonalnie) Generuj Tabelę Porównawczą Metryk ---
    # Możesz wczytać pliki evaluation_results.csv/.txt dla każdego modelu
    # i stworzyć zbiorczą tabelę porównawczą metryk.
    # ... (kod do wczytania i agregacji metryk) ...

    logging.info("--- Comparison Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare portfolio performance results against benchmarks.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file for the comparison script.")
    args = parser.parse_args()

    comparison_config = load_config(args.config)
    if comparison_config:
        main_comparison(comparison_config)

"""

**Wyjaśnienie Skryptu Porównawczego:**

1.  **Konfiguracja (`comparison_config.yaml`):** Skrypt wymaga pliku konfiguracyjnego YAML, który określa:
    *   `results_base_dir`: Główny katalog, w którym znajdują się podkatalogi z wynikami każdego modelu (np. `results/PortfolioCrossFormer`, `results/PortfolioMASTER`).
    *   `models_to_compare`: Lista nazw modeli (muszą odpowiadać nazwom podkatalogów i plików).
    *   `benchmark_data`: Ścieżka do pliku z danymi S&P 500 oraz nazwy kolumn z wartością indeksu i datą.
    *   `comparison_output_dir`: Katalog, w którym zostanie zapisany wynikowy wykres porównawczy i logi.
2.  **Ładowanie Danych Benchmarku:** Funkcja `load_benchmark_data` wczytuje dane S&P 500, konwertuje daty i wartości, oblicza dzienne zwroty i **kluczowe: oblicza skumulowaną krzywą wartości (`ValueCurve`)**, normalizując ją tak, by zaczynała się od 1.0.
3.  **Ładowanie Wyników Modeli:** Skrypt iteruje po liście `models_to_compare`, znajduje odpowiedni plik `{ModelName}_portfolio_value_curve.csv` w `results_base_dir`, wczytuje krzywą wartości za pomocą `load_portfolio_curve`. Określa wspólny zakres dat (`min_date`, `max_date`) na podstawie wczytanych krzywych modeli.
4.  **Przycinanie i Normalizacja:** Dane benchmarku oraz krzywe modeli są przycinane do wspólnego okresu (`min_date` do `max_date`). Następnie są **normalizowane**, aby *każda krzywa zaczynała się od wartości 1.0* na początku wspólnego okresu. To jest kluczowe dla wizualnego porównania wzrostu.
5.  **Generowanie Wykresu:** Tworzony jest wykres za pomocą Matplotlib:
    *   Krzywa benchmarku jest rysowana (np. linią przerywaną).
    *   Krzywe wartości dla każdego modelu są rysowane (różnymi kolorami).
    *   Dodawane są tytuły, etykiety, legenda.
    *   **Używana jest skala logarytmiczna (`yscale('log')`)**, która jest bardzo przydatna do wizualizacji wzrostu kapitału w długim okresie, ponieważ równe odległości na osi Y odpowiadają równym *procentowym* zmianom.
    *   Oś X jest formatowana jako daty.
    *   Wykres jest zapisywany do pliku PNG.
6.  **(Opcjonalnie) Tabela Porównawcza Metryk:** Skrypt można łatwo rozszerzyć, aby wczytywał również pliki `evaluation_results.csv` lub `.txt` dla każdego modelu i tworzył zbiorczą tabelę porównującą kluczowe metryki (CR, AR, SR, MDD, IC, ICIR, P@k) – podobną do tych, które generowaliśmy wcześniej w LaTeXu.

**Jak Uruchomić:**


1.  Zapisz powyższy kod jako `comparison_script.py`.
2.  Stwórz plik `comparison_config.yaml` i wypełnij go odpowiednimi ścieżkami i nazwami modeli.
3.  Uruchom z linii komend: `python comparison_script.py --config path/to/comparison_config.yaml`

Ten skrypt pozwoli Ci efektywnie porównać wyniki wszystkich Twoich modeli z benchmarkiem rynkowym i między sobą.
"""