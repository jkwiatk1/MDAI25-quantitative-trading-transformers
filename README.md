# Quantitative Trading with Transformer Models

Deep learning framework for portfolio optimization using transformer-based architectures on S&P 500 stock data.

## Overview

This project implements and compares multiple transformer architectures for predicting stock returns and optimizing portfolio allocation. The models predict future returns for multiple stocks simultaneously and select top-k assets for portfolio construction.

## Implemented Models

- **VanillaTransformer**: Standard transformer encoder architecture
- **iTransformer**: Inverted transformer treating stocks as tokens ([paper](https://arxiv.org/pdf/2310.06625))
- **CrossFormer**: Cross-dimensional transformer with segment-wise operations ([paper](https://arxiv.org/pdf/2108.00154))
- **TransformerCA**: Transformer with cross-attention mechanism
- **MASTER**: Multi-scale attention transformer ([paper](https://arxiv.org/abs/2312.15235))

## Project Structure

```
experiments/
  ├── models_grid_search_training_framework.py  # Hyperparameter grid search
  ├── models_evaluation_framework.py            # Model evaluation & metrics
  ├── models_comparison_script.py               # Compare models vs benchmarks
  ├── configs/                                  # YAML configuration files
  └── utils/                                    # Data loading, metrics, training
models/                                         # Model architectures
data/                                          # Financial data & results
```

## Usage

### 1. Grid Search Training
```bash
python -m experiments.models_grid_search_training_framework --config experiments/configs/training_config_iTransformer.yaml
```

### 2. Model Evaluation
```bash
python -m experiments.models_evaluation_framework
```
Evaluates all trained models and generates performance metrics.

### 3. Model Comparison
```bash
python -m experiments.models_comparison_script --config experiments/configs/comparison_config.yaml
```
Compares portfolio performance across models and benchmarks.

## Key Features

- **Rank Loss**: Custom loss function for portfolio ranking optimization
- **Portfolio Metrics**: Cumulative Return, Sharpe Ratio, Maximum Drawdown, Information Coefficient
- **Top-K Selection**: Dynamic portfolio allocation based on predicted returns
- **Benchmark Comparison**: S&P 500 Buy & Hold baseline

## Configuration

Edit YAML files in `experiments/configs/` to customize:
- Model hyperparameters (d_model, n_heads, dropout, etc.)
- Training settings (learning rate, batch size, epochs)
- Data parameters (lookback window, features, date range)
- Portfolio settings (top_k assets, risk-free rate)

## Requirements

- PyTorch
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- YAML
