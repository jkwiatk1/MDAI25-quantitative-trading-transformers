# Legacy Scripts

This folder contains older training and evaluation scripts that have been superseded by the unified framework.

## Archived Files

### Individual Model Training Scripts
- `CrossFormer_training.py` - Original CrossFormer training script
- `MASTER_training.py` - Original MASTER training script
- `models_training_framework.py` - Early training framework (before grid search version)
- `iTransformer_multi_finance_inputs.py` - iTransformer training script
- `iTransformer_multi_finance_inputs_grid_search.py` - iTransformer grid search script
- `Transformer_multi_finance_inputs.py` - Vanilla Transformer training script
- `Transformer_multi_finance_inputs_grid_search.py` - Vanilla Transformer grid search script

### Individual Model Evaluation Scripts
- `Portfolio_Selection_evaluation_CrossFormer.py`
- `Portfolio_Selection_evaluation_iTransformer.py`
- `Portfolio_Selection_evaluation_MASTER.py`
- `Portfolio_Selection_evaluation_TransformerCA.py`
- `Portfolio_Selection_evaluation_VanillaTransformer.py`

## Why Archived?

These scripts were replaced by a unified framework that handles all models consistently:

### Current Unified Framework (Use These Instead)
1. **models_grid_search_training_framework.py** - Hyperparameter tuning for all models
2. **models_evaluation_framework.py** - Evaluation for all models
3. **models_comparison_script.py** - Comparison across models and benchmarks

## Benefits of Unified Framework
- Single codebase for all models
- Consistent training/evaluation procedures
- Easier maintenance and bug fixes
- Configuration-driven (YAML files)
- Better code reusability

## When to Use Legacy Scripts
- Reference implementation details
- Debugging model-specific issues
- Historical comparisons
- Understanding evolution of the project

**Note**: These legacy scripts are kept for reference only and may not be maintained going forward.
