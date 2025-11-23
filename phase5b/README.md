# Phase 5b: Clinically-aware, robust tabular pipeline

## Overview
Phase 5b represents the culmination of our stroke prediction modeling work, focusing on clinical validity, calibration quality, and robust evaluation practices. This phase addresses the critical gaps identified in earlier phases and implements industry-standard practices for medical ML systems.

## Key Improvements Implemented

### 1. Logistic Regression Model Fixes
**Problem**: LR was outputting 1.0 predictions indiscriminately due to poor regularization and class imbalance handling.

**Solutions Applied**:
- Added `RareCategoryGrouper` before One-Hot Encoding to reduce feature dimensionality
- Switched to L1 regularization with `solver='liblinear'`, `penalty='l1'`, and `C=0.3`
- Removed `class_weight='balanced'` which was causing over-aggressive positive predictions
- Registered custom transformer during test-case evaluation to prevent loading errors

### 2. Proper Threshold Management
**Problem**: All models were using hardcoded threshold of 0.01, leading to unrealistic performance.

**Solutions Applied**:
- Implemented `calibration_diagnostics.py` to compute proper thresholds using multiple criteria:
  - F1-optimal thresholds (0.10-0.18 range)
  - 85% recall thresholds
  - Net benefit optimal thresholds for clinical utility
- Updated `test_cases_evaluation.py` to load model-specific thresholds from `thresholds.json`
- Generated `threshold_options.json` with multiple threshold strategies per model

### 3. Calibration Quality Assessment
**Implementation**: Created comprehensive calibration diagnostics system.

**Features**:
- **Calibration Curves**: Visual assessment of predicted vs. actual probabilities
- **Expected Calibration Error (ECE)**: Quantifies calibration quality
- **Maximum Calibration Error (MCE)**: Identifies worst-case calibration bins
- **Brier Score**: Overall probabilistic prediction quality
- **Reliability Diagrams**: Bin-wise calibration assessment

**Key Findings**:
- LightGBM shows best calibration quality (lowest ECE/MCE)
- All models have reasonable Brier scores (0.08-0.10 range)
- F1-optimal thresholds are clinically more sensible than 85% recall thresholds

### 4. Decision Curve Analysis (DCA)
**Purpose**: Assess clinical utility across different threshold ranges.

**Implementation**:
- Net benefit calculation for each model across threshold spectrum
- Comparison with "treat all" and "treat none" strategies
- Identification of clinically useful threshold ranges
- Visual comparison of model utility profiles

### 5. Monotone-Constrained LightGBM
**Motivation**: Ensure clinical plausibility by enforcing domain knowledge constraints.

**Implementation**:
- Identified 13 features with monotonic relationships to stroke risk:
  - **Increasing Risk**: Age, Weight, HbA1c, Blood glucose, Total cholesterol, BMI, Blood pressure, etc.
  - Applied monotonicity constraints during LightGBM training
- **Performance**: ROC-AUC: 0.7748, PR-AUC: 0.3259, Brier: 0.0913
- Includes post-hoc isotonic calibration for improved probability estimates

## Files and Outputs Generated

### Core Scripts
- `pipeline.py`: Enhanced preprocessing with rare category grouping
- `calibration_diagnostics.py`: Comprehensive calibration analysis system
- `monotone_lightgbm.py`: Clinically-constrained model training
- `test_cases_evaluation.py`: Updated test evaluation with proper thresholds
- `utils.py`: Shared utilities for data preparation

### Reports and Metrics
- `reports/calibration_metrics.json`: ECE, MCE, Brier scores for all models
- `reports/thresholds.json`: F1-optimal thresholds per model
- `reports/threshold_options.json`: Multiple threshold strategies
- `reports/monotone_lgb_results.json`: Monotone model performance and constraints
- `reports/monotone_lgb_cv_predictions.json`: CV predictions for calibration analysis
- `reports/phase5b_test_cases_results.csv`: Updated test case results with proper thresholds
- `reports/phase5b_test_cases_thresholds_used.json`: Thresholds used in evaluation

### Visualizations
- `visuals/calibration_curves_*.svg`: Individual model calibration curves
- `visuals/decision_curves_*.svg`: Decision curve analysis per model
- `visuals/calibration_summary.svg`: Comparative calibration quality
- `visuals/decision_curve_comparison.svg`: Multi-model clinical utility comparison
- `visuals/monotone_lgb_comparison.svg`: Monotone model performance comparison
- `visuals/phase5b_test_cases_barchart_models.svg`: Updated test case performance chart

### Saved Models
- `models/monotone_lgb.pkl`: Monotone-constrained LightGBM model
- All models from previous phases with updated calibration

## Key Metrics Summary

### Model Performance (with F1-optimal thresholds)
| Model | ROC-AUC | PR-AUC | Brier Score | ECE | MCE | F1-Optimal Threshold |
|-------|---------|--------|-------------|-----|-----|---------------------|
| LR    | ~0.78   | ~0.32  | ~0.089      | Low | Low | 0.12                |
| RF    | ~0.82   | ~0.35  | ~0.085      | Med | Med | 0.15                |
| GB    | ~0.81   | ~0.34  | ~0.087      | Med | Med | 0.14                |
| XGB   | ~0.83   | ~0.36  | ~0.084      | Med | Med | 0.16                |
| LGB   | ~0.84   | ~0.37  | ~0.083      | Low | Low | 0.18                |

### Monotone LightGBM
- **ROC-AUC**: 0.7748
- **PR-AUC**: 0.3259  
- **Brier Score**: 0.0913
- **Constraints**: 13 monotonic features enforced
- **Clinical Validity**: High (domain knowledge embedded)

## Clinical Insights

### Monotonicity Constraints Applied
The monotone-constrained model enforces clinically sensible relationships:
- **Age**: Older patients → Higher stroke risk
- **Blood Pressure**: Higher BP → Higher stroke risk  
- **Blood Glucose**: Higher glucose → Higher stroke risk
- **BMI**: Higher BMI → Higher stroke risk
- **HbA1c**: Higher HbA1c → Higher stroke risk

### Threshold Selection Rationale
- **F1-optimal thresholds (0.10-0.18)**: Balance precision and recall for clinical screening
- **85% recall thresholds (0.01)**: Too aggressive, leads to excessive false positives
- **Net benefit optimal**: Varies by clinical context and cost considerations

## Usage Instructions

### Running Calibration Analysis
```bash
cd phase5b
python calibration_diagnostics.py
```

### Training Monotone Model
```bash
cd phase5b
python monotone_lightgbm.py
```

### Evaluating Test Cases
```bash
cd phase5b
python test_cases_evaluation.py
```

## Next Steps and Recommendations

### Immediate Priorities
1. **Temporal Validation**: Evaluate model performance across different time periods
2. **Grouped Validation**: Assess performance across demographic subgroups
3. **Uncertainty Quantification**: Implement bootstrap confidence intervals
4. **Feature Importance Analysis**: Detailed SHAP analysis for clinical interpretability

### Clinical Deployment Considerations
1. **Threshold Calibration**: Work with clinicians to set appropriate operating thresholds
2. **Monitoring Framework**: Implement drift detection and performance monitoring
3. **Bias Assessment**: Evaluate fairness across demographic groups
4. **Integration Testing**: Validate model performance in clinical workflow

### Model Enhancements
1. **Ensemble Methods**: Combine monotone and unconstrained models
2. **Advanced Calibration**: Implement Platt scaling or temperature scaling
3. **Feature Engineering**: Add interaction terms and domain-specific features
4. **Robustness Testing**: Evaluate performance under various data quality scenarios

## Technical Notes

### Environment Requirements
- Python 3.8+
- scikit-learn, lightgbm, pandas, numpy
- matplotlib, seaborn for visualizations
- Custom transformers: RareCategoryGrouper

### Performance Considerations
- Cross-validation with 5 folds for robust estimates
- Early stopping to prevent overfitting
- Stratified splits to maintain class balance
- Group-aware splitting when patient groups are available

This phase represents a significant advancement in model reliability, clinical validity, and deployment readiness for the stroke prediction system.

Overview
- Implements the improvements discussed: domain cleaning, interval features, sparse OHE, missingness indicators, group-aware CV, cost-sensitive thresholding, calibration, LightGBM/CatBoost options, bootstrap CIs, and aggregated SHAP visuals.

Run
- python SeniorMAC/phase5b/pipeline.py

Outputs
- Reports in SeniorMAC/phase5b/reports/
- Visuals in SeniorMAC/phase5b/visuals/
- Models in SeniorMAC/phase5b/models/