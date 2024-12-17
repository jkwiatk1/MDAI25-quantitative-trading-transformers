# iTransformer for quantitative trading

## Basics:
* https://arxiv.org/pdf/2310.06625
* https://arxiv.org/pdf/2404.00424

## Running model:
```cmd
>path/transformer_vs_itransformer> python -m experiments.iTransformer_multi_finance_inputs --config=./experiments/configs/training_config.yaml
```

```cmd
>path/transformer_vs_itransformer> python -m experiments.iTransformer_multi
_finance_inputs --config=./experiments/configs/yahoo_training_config_iTransformer.yaml
```
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```