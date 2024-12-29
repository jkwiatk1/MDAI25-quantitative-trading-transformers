# iTransformer for quantitative trading

## Basics:
* https://arxiv.org/pdf/2310.06625
* https://arxiv.org/pdf/2404.00424

## Running model:
```cmd
>path/transformer_vs_itransformer> python -m experiments.Transformer_multi_finance_inputs --config=./experiments/configs/tick_10/yahoo_training_config_Transformer.yaml  
>path/transformer_vs_itransformer> python -m experiments.iTransformer_multi_finance_inputs --config=./experiments/configs/tick_100/yahoo_training_config_iTransformer.yaml  
```
**Grid search**
```cmd
>path/transformer_vs_itransformer> python -m experiments.iTransformer_multi_finance_inputs_grid_search --config=./experiments/configs/tick_10/yahoo_training_config_iTransformer_grid_search.yaml
>path/transformer_vs_itransformer> python -m experiments.Transformer_multi_finance_inputs_grid_search --config=./experiments/configs/tick_10/yahoo_training_config_Transformer_grid_search.yaml
```

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```