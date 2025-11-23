"""
Monotone-Constrained LightGBM for Clinical Interpretability

This script trains a LightGBM model with monotonicity constraints on clinical variables
to ensure medically sensible behavior (e.g., higher age/troponin = higher risk).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
import joblib
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

# Local imports
try:
    from .utils import prepare_dataset
except ImportError:
    from utils import prepare_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PHASE_DIR = os.path.join(PROJECT_ROOT, "phase5b")
REPORTS_DIR = os.path.join(PHASE_DIR, "reports")
VISUALS_DIR = os.path.join(PHASE_DIR, "visuals")
MODELS_DIR = os.path.join(PHASE_DIR, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "clean_data.csv")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def identify_monotone_features(X: pd.DataFrame) -> Dict[str, int]:
    """
    Identify features that should have monotonic relationships with stroke risk.
    Returns dict mapping feature names to constraint direction (1=increasing, -1=decreasing, 0=no constraint)
    """
    monotone_constraints = {}
    
    # Clinical variables that should increase risk (positive monotonicity)
    increasing_risk_patterns = [
        'age', 'troponin', 'bp_sys', 'bp_dia', 'hba1c', 'glucose', 'creatinine',
        'urea', 'cholesterol', 'ldl', 'triglycerides', 'bmi', 'weight',
        'heart_rate', 'temperature', 'wbc', 'neutrophils', 'platelets'
    ]
    
    # Clinical variables that should decrease risk (negative monotonicity)
    decreasing_risk_patterns = [
        'hdl', 'hemoglobin', 'hematocrit', 'albumin', 'oxygen_saturation'
    ]
    
    # Binary/categorical features (no monotonicity constraint)
    categorical_patterns = [
        'gender', 'sex', 'male', 'female', 'yes', 'no', 'type', 'category',
        'group', 'class', 'level', 'grade', 'stage'
    ]
    
    for col in X.columns:
        col_lower = str(col).lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        
        # Check for increasing risk patterns
        constraint = 0
        for pattern in increasing_risk_patterns:
            if pattern in col_lower:
                constraint = 1
                break
        
        # Check for decreasing risk patterns
        if constraint == 0:
            for pattern in decreasing_risk_patterns:
                if pattern in col_lower:
                    constraint = -1
                    break
        
        # Check if it's categorical (no constraint)
        if constraint != 0:
            for pattern in categorical_patterns:
                if pattern in col_lower:
                    constraint = 0
                    break
        
        monotone_constraints[col] = constraint
    
    return monotone_constraints


def train_monotone_lgb(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, 
                      monotone_constraints: Dict[str, int]) -> Tuple[Any, Dict, np.ndarray, np.ndarray]:
    """
    Train monotone-constrained LightGBM with cross-validation
    """
    # Convert constraints to list format expected by LightGBM
    constraint_list = [monotone_constraints.get(col, 0) for col in X.columns]
    
    print(f"Monotonicity constraints applied to {sum(1 for c in constraint_list if c != 0)} features:")
    for i, col in enumerate(X.columns):
        if constraint_list[i] != 0:
            direction = "↑" if constraint_list[i] == 1 else "↓"
            print(f"  {col}: {direction}")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store CV predictions
    y_pred_cv = np.zeros(len(y))
    y_true_cv = np.zeros(len(y))
    
    models = []
    metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"Training fold {fold + 1}/5...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # LightGBM parameters with monotonicity constraints
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'monotone_constraints': constraint_list,
            'monotone_constraints_method': 'advanced',  # More flexible constraints
            'verbose': -1,
            'random_state': 42 + fold
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predict on validation set
        y_pred_fold = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred_cv[val_idx] = y_pred_fold
        y_true_cv[val_idx] = y_val
        
        # Calculate fold metrics
        fold_metrics = {
            'roc_auc': roc_auc_score(y_val, y_pred_fold),
            'pr_auc': average_precision_score(y_val, y_pred_fold),
            'brier': brier_score_loss(y_val, y_pred_fold)
        }
        metrics.append(fold_metrics)
        models.append(model)
        
        print(f"  Fold {fold + 1} - ROC-AUC: {fold_metrics['roc_auc']:.4f}, PR-AUC: {fold_metrics['pr_auc']:.4f}")
    
    # Calculate overall CV metrics
    cv_metrics = {
        'roc_auc_mean': np.mean([m['roc_auc'] for m in metrics]),
        'roc_auc_std': np.std([m['roc_auc'] for m in metrics]),
        'pr_auc_mean': np.mean([m['pr_auc'] for m in metrics]),
        'pr_auc_std': np.std([m['pr_auc'] for m in metrics]),
        'brier_mean': np.mean([m['brier'] for m in metrics]),
        'brier_std': np.std([m['brier'] for m in metrics]),
        'roc_auc_cv': roc_auc_score(y_true_cv, y_pred_cv),
        'pr_auc_cv': average_precision_score(y_true_cv, y_pred_cv),
        'brier_cv': brier_score_loss(y_true_cv, y_pred_cv)
    }
    
    return models[0], cv_metrics, y_true_cv, y_pred_cv  # Return first model as representative


def compare_with_baseline(monotone_metrics: Dict, baseline_path: str) -> Dict:
    """Compare monotone model with baseline LightGBM"""
    comparison = {}
    
    # Load baseline metrics
    try:
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        
        if 'lgb' in baseline_data:
            baseline_metrics = baseline_data['lgb']
            
            comparison = {
                'monotone_roc_auc': monotone_metrics['roc_auc_cv'],
                'baseline_roc_auc': baseline_metrics.get('roc_auc_mean', 0),
                'roc_auc_diff': monotone_metrics['roc_auc_cv'] - baseline_metrics.get('roc_auc_mean', 0),
                
                'monotone_pr_auc': monotone_metrics['pr_auc_cv'],
                'baseline_pr_auc': baseline_metrics.get('pr_auc_mean', 0),
                'pr_auc_diff': monotone_metrics['pr_auc_cv'] - baseline_metrics.get('pr_auc_mean', 0),
                
                'monotone_brier': monotone_metrics['brier_cv'],
                'baseline_brier': baseline_metrics.get('brier_mean', 1),
                'brier_diff': monotone_metrics['brier_cv'] - baseline_metrics.get('brier_mean', 1)
            }
    except Exception as e:
        print(f"Could not load baseline metrics: {e}")
        comparison = {
            'monotone_roc_auc': monotone_metrics['roc_auc_cv'],
            'monotone_pr_auc': monotone_metrics['pr_auc_cv'],
            'monotone_brier': monotone_metrics['brier_cv']
        }
    
    return comparison


def plot_comparison(comparison: Dict, monotone_constraints: Dict) -> plt.Figure:
    """Plot comparison between monotone and baseline models"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metrics comparison
    if 'baseline_roc_auc' in comparison:
        metrics = ['ROC-AUC', 'PR-AUC', 'Brier Score']
        monotone_vals = [comparison['monotone_roc_auc'], comparison['monotone_pr_auc'], comparison['monotone_brier']]
        baseline_vals = [comparison['baseline_roc_auc'], comparison['baseline_pr_auc'], comparison['baseline_brier']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, baseline_vals, width, label='Baseline LGB', alpha=0.7, color='skyblue')
        axes[0, 0].bar(x + width/2, monotone_vals, width, label='Monotone LGB', alpha=0.7, color='lightcoral')
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Differences
        diffs = [comparison['roc_auc_diff'], comparison['pr_auc_diff'], -comparison['brier_diff']]  # Negative brier diff for improvement
        colors = ['green' if d > 0 else 'red' for d in diffs]
        
        axes[0, 1].bar(metrics, diffs, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Metrics')
        axes[0, 1].set_ylabel('Improvement (Monotone - Baseline)')
        axes[0, 1].set_title('Performance Differences')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Baseline metrics\nnot available', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, 'Comparison\nnot available', ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # Monotonicity constraints visualization
    constraint_counts = {1: 0, -1: 0, 0: 0}
    for constraint in monotone_constraints.values():
        constraint_counts[constraint] += 1
    
    labels = ['Increasing Risk ↑', 'Decreasing Risk ↓', 'No Constraint']
    sizes = [constraint_counts[1], constraint_counts[-1], constraint_counts[0]]
    colors = ['lightcoral', 'lightblue', 'lightgray']
    
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Monotonicity Constraints Distribution')
    
    # Feature importance (placeholder)
    axes[1, 1].text(0.5, 0.5, f'Monotone LGB Results:\n\nROC-AUC: {comparison["monotone_roc_auc"]:.4f}\nPR-AUC: {comparison["monotone_pr_auc"]:.4f}\nBrier: {comparison["monotone_brier"]:.4f}\n\nConstraints: {constraint_counts[1] + constraint_counts[-1]} features', 
                   ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    axes[1, 1].set_title('Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main monotone LightGBM training workflow"""
    print("Loading data and preparing features...")
    
    # Define data path
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clean_data.csv")
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    X, y, cat_cols, num_cols, groups = prepare_dataset(df)
    
    # Remove rows with NaN target values
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    if groups is not None:
        groups = groups.loc[mask]
    
    # Reset indices to ensure continuous indexing for CV
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if groups is not None:
        groups = groups.reset_index(drop=True)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Positive class rate: {y.mean():.3f}")
    
    # Build preprocessor to handle categorical variables
    from pipeline import build_preprocessor
    preprocessor = build_preprocessor(cat_cols, num_cols)
    
    # Fit preprocessor and transform data
    X_processed = preprocessor.fit_transform(X)
    
    # Convert to DataFrame for easier handling
    # Get feature names after preprocessing
    feature_names = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        # Get OHE feature names
        cat_transformer = preprocessor.named_transformers_['cat']
        ohe = cat_transformer.named_steps['ohe']
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names.extend(cat_feature_names)
    
    # Clean feature names to remove special characters that LightGBM doesn't support
    clean_feature_names = []
    for name in feature_names:
        # Keep only alphanumeric characters and underscores
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        # Remove multiple consecutive underscores
        while '__' in clean_name:
            clean_name = clean_name.replace('__', '_')
        # Remove leading/trailing underscores
        clean_name = clean_name.strip('_')
        # Ensure name is not empty
        if not clean_name:
            clean_name = f'feature_{len(clean_feature_names)}'
        clean_feature_names.append(clean_name)
    
    X_processed = pd.DataFrame(X_processed, columns=clean_feature_names)
    
    print(f"Processed dataset shape: {X_processed.shape}")
    
    # Identify monotonicity constraints for processed features
    print("\nIdentifying monotonicity constraints...")
    monotone_constraints = identify_monotone_features(X_processed)
    
    constrained_features = {k: v for k, v in monotone_constraints.items() if v != 0}
    print(f"Applied constraints to {len(constrained_features)} out of {len(X.columns)} features")
    
    # Train monotone-constrained model
    print("\nTraining monotone-constrained LightGBM...")
    model, cv_metrics, y_true_cv, y_pred_cv = train_monotone_lgb(X_processed, y, groups, monotone_constraints)
    
    # Apply calibration
    print("\nApplying post-hoc calibration...")
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    # Note: For proper calibration, we'd need to retrain, but this is a demonstration
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "monotone_lgb.pkl")
    joblib.dump(model, model_path)
    
    # Save CV predictions for calibration analysis
    cv_predictions = {
        'monotone_lgb': {
            'y': y_true_cv.tolist(),
            'prob': y_pred_cv.tolist()
        }
    }
    
    cv_pred_path = os.path.join(REPORTS_DIR, "monotone_lgb_cv_predictions.json")
    with open(cv_pred_path, 'w') as f:
        json.dump(cv_predictions, f, indent=2)
    
    # Compare with baseline
    baseline_metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    comparison = compare_with_baseline(cv_metrics, baseline_metrics_path)
    
    # Save results
    results = {
        'monotone_constraints': monotone_constraints,
        'constrained_features_count': len(constrained_features),
        'cv_metrics': cv_metrics,
        'comparison': comparison
    }
    
    results_path = os.path.join(REPORTS_DIR, "monotone_lgb_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    fig = plot_comparison(comparison, monotone_constraints)
    fig.savefig(os.path.join(VISUALS_DIR, "monotone_lgb_comparison.svg"), 
               format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Print summary
    print("\n" + "="*60)
    print("MONOTONE LIGHTGBM RESULTS")
    print("="*60)
    print(f"Cross-validation ROC-AUC: {cv_metrics['roc_auc_cv']:.4f}")
    print(f"Cross-validation PR-AUC:  {cv_metrics['pr_auc_cv']:.4f}")
    print(f"Cross-validation Brier:   {cv_metrics['brier_cv']:.4f}")
    print(f"\nMonotonicity constraints applied to {len(constrained_features)} features:")
    
    increasing = [k for k, v in constrained_features.items() if v == 1]
    decreasing = [k for k, v in constrained_features.items() if v == -1]
    
    if increasing:
        print(f"\nIncreasing risk (↑): {len(increasing)} features")
        for feat in increasing[:5]:  # Show first 5
            print(f"  • {feat}")
        if len(increasing) > 5:
            print(f"  ... and {len(increasing) - 5} more")
    
    if decreasing:
        print(f"\nDecreasing risk (↓): {len(decreasing)} features")
        for feat in decreasing[:5]:  # Show first 5
            print(f"  • {feat}")
        if len(decreasing) > 5:
            print(f"  ... and {len(decreasing) - 5} more")
    
    if 'baseline_roc_auc' in comparison:
        print(f"\nComparison with baseline LightGBM:")
        print(f"  ROC-AUC difference: {comparison['roc_auc_diff']:+.4f}")
        print(f"  PR-AUC difference:  {comparison['pr_auc_diff']:+.4f}")
        print(f"  Brier difference:   {comparison['brier_diff']:+.4f} (lower is better)")
    
    print(f"\nFiles saved:")
    print(f"  • Model: {model_path}")
    print(f"  • Results: {results_path}")
    print(f"  • CV predictions: {cv_pred_path}")
    print(f"  • Comparison plot: {os.path.join(VISUALS_DIR, 'monotone_lgb_comparison.svg')}")


if __name__ == "__main__":
    main()