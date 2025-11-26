# Phase 2: Baseline Models & Preprocessing

## Overview
This phase implements baseline machine learning models with comprehensive preprocessing strategies for mortality risk prediction in stroke patients. The focus is on establishing solid baselines using CatBoost with and without SMOTE to handle class imbalance.

## Key Components

### 1. Data Preprocessing
- **Missing Value Imputation**: KNN Imputation (k=5) for robust handling of missing data
- **Categorical Encoding**: Label encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Class Imbalance**: SMOTE oversampling to address severe class imbalance

### 2. Baseline Models
- **CatBoost (Original Data)**: Gradient boosting on original imbalanced dataset
- **CatBoost (SMOTE Data)**: Gradient boosting on SMOTE-balanced dataset

### 3. Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve for binary classification
- **Confusion Matrix**: Detailed classification performance
- **Precision-Recall Curves**: Performance across different thresholds

## File Structure
```
phase2_baseline_models/
├── README.md                           # This documentation
├── phase2_baseline_pipeline.py         # Main pipeline script
├── models/                             # Saved models and preprocessors
│   ├── phase2_best_model.cbm          # Best performing CatBoost model
│   └── phase2_preprocessors.pkl        # Preprocessing objects
├── visuals/                            # Generated visualizations
│   ├── target_distribution.png         # Class distribution analysis
│   ├── smote_comparison.png            # Before/after SMOTE comparison
│   ├── model_performance_comparison.png # Model metrics comparison
│   ├── roc_curves_comparison.png       # ROC curves for both models
│   ├── precision_recall_curves.png     # Precision-recall analysis
│   ├── confusion_matrices.png          # Confusion matrices
│   └── feature_importance.png          # Top feature importances
└── reports/                            # Analysis reports
    ├── phase2_results_summary.json     # Structured results data
    └── phase2_report.md                # Comprehensive analysis report
```

## Usage

### Running the Pipeline
```bash
cd phase2_baseline_models
python phase2_baseline_pipeline.py
```

### Expected Outputs
1. **Models**: Trained CatBoost models saved in `models/` directory
2. **Visualizations**: Comprehensive plots in `visuals/` directory
3. **Reports**: Detailed analysis in `reports/` directory

## Key Features

### Preprocessing Pipeline
- Handles mixed data types (numerical and categorical)
- Robust missing value imputation using KNN
- Proper train/test splitting with stratification
- Feature scaling for optimal model performance

### Class Imbalance Handling
- SMOTE (Synthetic Minority Oversampling Technique)
- Comparison between original and balanced datasets
- Visualization of class distribution changes

### Model Training
- CatBoost with optimized hyperparameters
- Early stopping to prevent overfitting
- Cross-validation for robust evaluation
- Feature importance analysis

### Comprehensive Evaluation
- Multiple performance metrics
- ROC and Precision-Recall curve analysis
- Confusion matrix visualization
- Model comparison framework

## Expected Results

### Performance Metrics
- **Baseline Accuracy**: 85-95% (depending on class imbalance handling)
- **F1-Score**: 0.3-0.7 (challenging due to severe class imbalance)
- **AUC-ROC**: 0.7-0.9 (good discrimination capability)

### Key Insights
1. **Class Imbalance Impact**: Severe imbalance (95:5 ratio) significantly affects model performance
2. **SMOTE Effectiveness**: May improve recall but could reduce precision
3. **Feature Importance**: Age, hypertension, and heart disease likely to be top predictors
4. **Model Selection**: Best model chosen based on AUC-ROC for medical applications

## Next Steps (Phase 3)
- Advanced ML models (Random Forest, XGBoost, LightGBM)
- Deep learning approaches (Neural Networks)
- Ensemble methods and model stacking
- Advanced feature engineering
- Hyperparameter optimization

## Dependencies
- pandas, numpy: Data manipulation
- scikit-learn: ML algorithms and preprocessing
- imbalanced-learn: SMOTE implementation
- catboost: Gradient boosting framework
- matplotlib, seaborn: Visualization
- pickle, json: Model and results serialization

## Notes
- All visualizations are saved as high-resolution PNG files
- Models are saved in CatBoost's native format for optimal loading
- Preprocessing objects are pickled for consistent transformation
- Results are structured in JSON format for easy analysis


## Augmentation: Before vs After (Phase 2)

This section documents augmentation experiments for Phase 2, including class imbalance oversampling and missingness-aware augmentation. It captures what we tried, challenges, outputs, and how to reproduce.

### How to Reproduce
- cd phase2_baseline_models
- python generate_augmentation_visuals.py

### Inputs & Outputs
- Visuals saved to visuals/augmentation/
  - auc_by_technique.png
  - class_distribution_comparison.png
  - confusion_matrices_none_vs_smote.png
  - feature_shift_checks.png
  - metrics_bar_death_class.png
  - missingness_heatmap.png
  - missingness_rates.png
  - roc_curves_comparison.png
- Reports saved to reports/augmentation/
  - metrics_by_technique.json
  - augmentation_summary.md

### Techniques Compared
- none: No augmentation
- random_over, smote, bsmote, adasyn: Class-imbalance oversampling baselines
- hba1c_aug: Adds an HbA1c-missing indicator and conditional synthetic values for training rows with missing HbA1c (train-only)
- multi_impute_aug: Train-only multiple imputation draws (IterativeImputer) stacked to create plausible training copies; test set untouched; indicators retained

### Summary of Results (current run)
- Baseline AUC (none): ~0.811
- Missingness-focused:
  - hba1c_aug AUC: ~0.782
  - multi_impute_aug AUC: ~0.792
- Oversampling baselines:
  - smote AUC: ~0.801, adasyn AUC: ~0.815, bsmote AUC: ~0.819, random_over AUC: ~0.773
- Note: Recall remained low at default threshold; consider PR/threshold sweeps and calibration for better recall/precision balance.

### Challenges & Mitigations
- Severe class imbalance: Evaluated multiple oversamplers; documented prevalence shifts in class_distribution_comparison.png
- Extreme missingness (e.g., HbA1c ~94% missing):
  - Added train-only missingness indicators
  - Implemented train-only multiple imputation augmentation and HbA1c-focused conditional sampling
  - Strict leakage prevention: all imputers/encoders fit on training only; test untouched
- Distribution shift risk from oversampling: feature_shift_checks.png used for sanity checks; gains kept modest

### Clinical Notes
- HbA1c augmentation provides transparency via a missingness indicator and conservative conditional sampling; given high missingness, we rely more on the indicator than the synthetic value for signal
- When augmentation yields marginal AUC gains, prefer simpler baselines or use calibration rather than aggressive synthetic balancing depending on clinical context

### Traceability
- Script: generate_augmentation_visuals.py
- Artifacts: visuals/augmentation/* and reports/augmentation/*


## Missingness: Before vs After (Phase 2)

This section captures the focused missingness handling experiment that mirrors our baseline pipeline and isolates the impact of missing data strategies.

### How to Reproduce
- cd phase2_baseline_models
- python analyze_missingness_before_after.py

### What We Compare
- BEFORE: Median imputation, no missingness indicators
- AFTER: Add is_missing__* indicators + KNNImputer (numeric)

### Key Results (default threshold 0.5)
- Acc, F1, ROC-AUC, AUPRC are written to reports/missingness/missingness_metrics.json and summarized in reports/missingness/missingness_before_after.md
- ROC/PR curves and confusion matrices are in visuals/missingness/

### Threshold Optimization
- We sweep thresholds to: (a) maximize F1; (b) maintain Precision ≥ 0.8 with highest Recall
- Generated artifacts:
  - visuals/missingness/threshold_sweep_f1.png
  - visuals/missingness/threshold_sweep_precision_recall.png
  - visuals/missingness/confusion_opt_f1.png (confusions @ best-F1 thresholds)
  - visuals/missingness/confusion_p80.png (confusions @ Precision≥0.8 thresholds)
- Chosen thresholds and scores are logged in reports/missingness/missingness_before_after.md and reports/missingness/missingness_metrics.json

### Indicator Importance Snapshot
- We compute feature importances from the AFTER model and surface the top is_missing__* indicators
- Artifact: visuals/missingness/indicator_importance.png
- The top indicators are also listed in reports/missingness/missingness_before_after.md

### HbA1c Missingness Rates
- If an HbA1c column exists, its train/test missingness rates are logged to the report for traceability

### Artifacts
- Visuals: visuals/missingness/
- Reports: reports/missingness/

### Notes
- No changes were made to augmentation code; this is a clean before/after comparison focused solely on missingness handling.


## Overfitting Diagnostics (Phase 2)

This section summarizes overfitting checks computed during the Phase 2 baseline pipeline and where to find the artifacts.

- Artifacts:
  - Report JSON: reports/phase2_overfitting.json
- Criteria:
  - f1_train_minus_test_threshold = 0.05
  - auc_train_minus_test_threshold = 0.05
  - test_vs_cv_z_threshold = 2.0
- Current Summary (latest run):
  - catboost_weighted: pass (train_f1=1.00, test_f1=1.00, f1_gap=0.00; train_auc=1.00, test_auc=1.00, auc_gap=0.00; cv_f1_mean=1.00, cv_f1_std=0.00, z=0.00)
  - catboost_standard: pass (train_f1=1.00, test_f1=1.00, f1_gap=0.00; train_auc=1.00, test_auc=1.00, auc_gap=0.00; cv_f1_mean=1.00, cv_f1_std=0.00, z=0.00)
- Quick commands:
  - cd phase2_baseline_models
  - python phase2_baseline_pipeline.py  # regenerates models and reports including phase2_overfitting.json

Notes
- The overfitting report is generated automatically at the end of the pipeline run.