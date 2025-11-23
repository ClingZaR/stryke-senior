# Phase 1: Data Exploration & Visualization

## Overview
This is the foundational phase of our mortality risk prediction project. Phase 1 focuses on understanding the dataset structure, identifying patterns, and establishing baseline insights that will guide all subsequent modeling phases.

## Objectives
- **Data Understanding**: Comprehensive exploration of the stroke mortality dataset
- **Missing Value Analysis**: Identify patterns and extent of missing data
- **Class Imbalance Assessment**: Understand the distribution of mortality outcomes
- **Feature Correlation Analysis**: Discover relationships between variables and mortality
- **Data Quality Assessment**: Identify outliers, inconsistencies, and data quality issues
- **Visual Insights**: Create informative visualizations for stakeholder communication

## Key Deliverables
1. **Comprehensive EDA Report**: Statistical summaries and data profiling
2. **Correlation Heatmaps**: Feature relationships with mortality outcome
3. **Missing Value Visualizations**: Patterns and impact assessment
4. **Distribution Plots**: Understanding feature distributions across mortality groups
5. **Class Imbalance Analysis**: Mortality vs survival rates breakdown

## Methodology
- **Descriptive Statistics**: Mean, median, mode, standard deviation for all features
- **Correlation Analysis**: Pearson and Spearman correlations with mortality outcome
- **Missing Data Profiling**: MCAR, MAR, MNAR pattern identification
- **Outlier Detection**: Statistical and visual outlier identification
- **Categorical Analysis**: Frequency distributions and chi-square tests
- **Temporal Analysis**: Time-based patterns in admissions and outcomes

## Key Insights Expected
- Which clinical features are most correlated with mortality?
- What is the extent and pattern of missing data?
- Are there demographic disparities in mortality rates?
- What is the class imbalance ratio (death vs survival)?
- Are there any data quality issues that need addressing?

## Difference from Future Phases
**Phase 1** is purely exploratory - no modeling or prediction occurs here. This phase establishes the foundation for:
- **Phase 2**: Will use these insights to design preprocessing strategies
- **Phase 3**: Will leverage correlation findings for feature engineering
- **Phase 4**: Will use data quality insights for optimization strategies

## Files in This Phase
- `phase1_comprehensive_eda.py`: Main analysis script with all visualizations
- `visuals/`: Directory containing all generated plots and charts
- `README.md`: This documentation file

## Expected Runtime
Approximately 2-3 minutes for complete analysis on the stroke dataset.

## Dependencies
- pandas, numpy: Data manipulation
- matplotlib, seaborn, plotly: Visualization
- scipy: Statistical analysis
- missingno: Missing data visualization

---
*This phase sets the foundation for building a robust, clinically-relevant mortality prediction model.*

## Challenges Faced & Improvements (Phase 1)

### Key Challenges
- Severe class imbalance observed (death vs survival highly skewed), indicating downstream sensitivity to recall at default thresholds and the need for imbalance methods in Phase 2+.
- Extreme missingness in certain labs (e.g., HbA1c with very high missing rate), prompting careful handling strategies and missingness indicators in later phases.
- Mixed data types with potential categorical inconsistencies (e.g., string variants), requiring normalization before modeling.
- Potential target leakage candidates identified via correlation and domain review (e.g., post-admission artifacts), flagged to avoid in modeling.

### Actions/Decisions Taken
- Produced comprehensive missingness profiling (heatmaps and rates) to guide imputation plans.
- Established the need for train-only imputation and missingness indicators for highly-missing features in Phase 2.
- Documented class imbalance baselines to compare against oversampling techniques (SMOTE family) later.
- Standardized categorical values and established consistent encodings to be reused in downstream phases.

### Outputs Referenced
- Visuals: see visuals/ for correlation_matrix.png and feature_distributions.png
- Script: phase1_comprehensive_eda.py

### What Carried Forward
- Use missingness indicators for top-missing features (implemented in Phase 2 augmentation experiments).
- Evaluate oversampling vs baseline and assess calibration/threshold tuning (Phase 2 and 3).
- Keep strict leakage prevention in all preprocessing (fit on train only).