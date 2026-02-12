# Tuned model training suite

This folder recreates **all model families used in the project** and adds explicit hyperparameter tuning.

## Models covered
- Logistic Regression
- Naive Bayes
- Random Forest
- XGBoost
- LightGBM
- BiLSTM
- DistilBERT
- TextGCN (enhanced variant)

## Outputs
- Models: `models/tuned/`
- Metrics: `reports/tuned/`

## Quick start
```bash
# run all tuned models with explicit per-model progress
python tuned_models/run_all_tuned.py

# skip models that are already trained
python tuned_models/run_all_tuned.py --skip-completed

# run only selected tuned models
python tuned_models/run_all_tuned.py --models tune_classical_models tune_bilstm

# keep going even if one model fails
python tuned_models/run_all_tuned.py --continue-on-error
```


## Notes
- Tuning uses EMSCAD train/validation splits from `data/processed/splits.json`.
- Best validation-F1 configuration is selected for each model, then evaluated on EMSCAD test + OpenDataBay.
- Deep models are intentionally tuned with compact search spaces so the process stays practical on local machines.
- The runner prints `[current/total]` status and elapsed time so long jobs no longer look frozen.
- `--skip-completed` checks for expected artifacts in `models/tuned/` before launching each trainer.
