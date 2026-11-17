# Unified Predictive Maintenance (AI4I 2020)

Binary machine-failure prediction with Logistic Regression, Random Forest and a Keras ANN, including a reliability audit,
a cost-based threshold analysis, a controlled ANN experiment (learning rate x 5 seeds), two extra experiments
(SMOTE-NC oversampling and XGBoost) and business recommendations.

## Dataset
AI4I 2020 Predictive Maintenance Dataset (UCI): https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
Place `ai4i2020.csv` in `data/` (or next to the notebook).

## Structure
```
notebooks/Predictive_Maintenance.ipynb   main notebook (run top to bottom)
results/model_comparison.csv             test metrics per model (threshold 0.5 and cost-optimal)
results/experiment_results.csv           learning-rate experiment (15 ANN runs) + 5 Random Forest reference runs
results/xgboost_experiment_results.csv   XGBoost runs (5 seeds) with the selected settings
reports/Final_Technical_Report.pdf       (to be added)
requirements.txt
```

## How to run
```
pip install -r requirements.txt
jupyter notebook notebooks/Predictive_Maintenance.ipynb     # Run All (about 35-40 min on CPU, mostly the repeated ANN runs)
```

## Key assumptions
- `TWF, HDF, PWF, OSF, RNF`, `UDI` and `Product ID` are excluded from the features (leakage safeguard).
- Stratified 64/16/20 train/validation/test split; scaling and encoding fitted on the training set only; tuning, early stopping and threshold selection use the validation set only.
- Inspection cost (500) and missed-failure cost (10,000) are HYPOTHETICAL.
- Random seeds: 42 (main pipeline), 11/22/33/44/55 (repeated runs). Results can vary slightly with other library versions/hardware.
- Tested with Python 3.12, TensorFlow 2.21, scikit-learn 1.9, xgboost 3.4, imbalanced-learn.
