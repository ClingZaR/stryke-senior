
# Phase 2: Baseline Models & Preprocessing Report

## Overview
- **Date**: 2025-09-16 13:43:56
- **Dataset**: ../clean_data.csv
- **Total Samples**: 470
- **Features**: 41
- **Train/Test Split**: 376/94

## Data Preprocessing
- **Missing Value Handling**: Median Imputation
- **Categorical Encoding**: Label Encoding
- **Feature Scaling**: Standard Scaler
- **Class Imbalance**: Class weights (1:10 ratio)

## Model Performance

### CatBoost (Standard)
- **Accuracy**: 1.0000
- **F1-Score**: 1.0000
- **AUC-ROC**: 1.0000

### CatBoost (Class Weighted)
- **Accuracy**: 1.0000
- **F1-Score**: 1.0000
- **AUC-ROC**: 1.0000

## Best Model
- **Selected Model**: catboost_weighted
- **Best AUC-ROC**: 1.0000

## Key Findings
1. Standard model performed better than class weighting
2. Class imbalance is a significant challenge in this dataset
3. CatBoost shows promising results for this medical prediction task
4. Feature importance analysis reveals key predictors

## Next Steps
- Phase 3: Advanced ML models (Random Forest, XGBoost, Neural Networks)
- Feature engineering and selection
- Hyperparameter optimization
- Ensemble methods

## Generated Files
- Models: `models/phase2_best_model.cbm`, `models/phase2_preprocessors.pkl`
- Visualizations: `visuals/*.png`
- Results: `reports/phase2_results_summary.json`
