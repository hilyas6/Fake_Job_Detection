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
python tuned_models/run_all_tuned.py
```

## Notes
- Tuning uses EMSCAD train/validation splits from `data/processed/splits.json`.
- Best validation-F1 configuration is selected for each model, then evaluated on EMSCAD test + OpenDataBay.
- Deep models are intentionally tuned with compact search spaces so the process stays practical on local machines.
