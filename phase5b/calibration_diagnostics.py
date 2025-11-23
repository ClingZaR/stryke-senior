"""
Calibration Diagnostics and Threshold Optimization for Phase 5b Models

This script generates:
1. Calibration curves with ECE/MCE/Brier scores
2. Proper threshold selection based on F1, recall, and net benefit
3. Decision curve analysis for clinical utility
4. Visual comparisons of calibration quality across models
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as CalibrationLR
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


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Maximum Calibration Error (MCE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def net_benefit(y_true: np.ndarray, y_pred: np.ndarray, threshold_prob: float) -> float:
    """Compute net benefit for decision curve analysis"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    
    # Net benefit = (TP/n) - (FP/n) * (threshold/(1-threshold))
    if threshold_prob >= 1.0:
        return 0.0
    
    weight = threshold_prob / (1 - threshold_prob)
    return (tp / n) - (fp / n) * weight


def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Find optimal thresholds using different criteria"""
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # F1-based threshold
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    f1_threshold = pr_thresholds[min(best_f1_idx, len(pr_thresholds) - 1)]
    
    # High recall threshold (85% recall)
    recall_85_threshold = 0.01
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        if np.sum(pred) > 0:
            current_recall = np.sum((pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
            if current_recall >= 0.85:
                recall_85_threshold = t
                break
    
    # Net benefit optimal threshold
    net_benefits = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        nb = net_benefit(y_true, pred, t)
        net_benefits.append(nb)
    
    best_nb_idx = np.argmax(net_benefits)
    net_benefit_threshold = thresholds[best_nb_idx]
    
    return {
        'f1_optimal': float(f1_threshold),
        'recall_85': float(recall_85_threshold),
        'net_benefit_optimal': float(net_benefit_threshold)
    }


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, 
                          ece: float, mce: float, brier: float) -> plt.Figure:
    """Plot calibration curve with reliability diagram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy='uniform'
    )
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-', 
             label=f'{model_name}\nECE: {ece:.3f}\nMCE: {mce:.3f}\nBrier: {brier:.3f}')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'Calibration Curve - {model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reliability histogram
    ax2.hist(y_prob, bins=20, alpha=0.7, density=True, label='Predicted probabilities')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Probability Distribution - {model_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_decision_curves(results: Dict[str, Dict]) -> plt.Figure:
    """Plot decision curves for all models"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for model_name, data in results.items():
        if 'y_true' in data and 'y_prob' in data:
            y_true, y_prob = data['y_true'], data['y_prob']
            net_benefits = []
            
            for t in thresholds:
                pred = (y_prob >= t).astype(int)
                nb = net_benefit(y_true, pred, t)
                net_benefits.append(nb)
            
            ax.plot(thresholds, net_benefits, label=f'{model_name.upper()}', linewidth=2)
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Treat None')
    
    # Treat all line
    if results:
        sample_data = next(iter(results.values()))
        if 'y_true' in sample_data:
            y_true = sample_data['y_true']
            prevalence = np.mean(y_true)
            treat_all_nb = [prevalence - t/(1-t) * (1-prevalence) if t < 1 else 0 for t in thresholds]
            ax.plot(thresholds, treat_all_nb, color='gray', linestyle='--', alpha=0.7, label='Treat All')
    
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    return fig


def main():
    """Main calibration diagnostics workflow"""
    print("Loading CV predictions and computing calibration diagnostics...")
    
    # Load CV predictions
    cv_path = os.path.join(REPORTS_DIR, "cv_predictions.json")
    if not os.path.exists(cv_path):
        print(f"CV predictions not found at {cv_path}")
        return
    
    with open(cv_path, 'r') as f:
        cv_data = json.load(f)
    
    results = {}
    calibration_metrics = {}
    optimal_thresholds = {}
    
    # Process each model
    for model_name in ['lr', 'rf', 'gb', 'xgb', 'lgb']:
        if model_name in cv_data:
            print(f"Processing {model_name.upper()}...")
            
            y_true = np.array(cv_data[model_name]['y'])
            y_prob = np.array(cv_data[model_name]['prob'])
            
            # Compute calibration metrics
            ece = expected_calibration_error(y_true, y_prob)
            mce = maximum_calibration_error(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            
            calibration_metrics[model_name] = {
                'ECE': float(ece),
                'MCE': float(mce),
                'Brier': float(brier)
            }
            
            # Find optimal thresholds
            thresholds = find_optimal_thresholds(y_true, y_prob)
            optimal_thresholds[model_name] = thresholds
            
            # Store for decision curves
            results[model_name] = {
                'y_true': y_true,
                'y_prob': y_prob,
                'ece': ece,
                'mce': mce,
                'brier': brier
            }
            
            # Plot calibration curve
            fig = plot_calibration_curve(y_true, y_prob, model_name.upper(), ece, mce, brier)
            fig.savefig(os.path.join(VISUALS_DIR, f"calibration_curve_{model_name}.svg"), 
                       format='svg', bbox_inches='tight', dpi=300)
            plt.close(fig)
    
    # Plot decision curves
    if results:
        fig = plot_decision_curves(results)
        fig.savefig(os.path.join(VISUALS_DIR, "decision_curves.svg"), 
                   format='svg', bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Create summary calibration plot
    if calibration_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = list(calibration_metrics.keys())
        ece_values = [calibration_metrics[m]['ECE'] for m in models]
        mce_values = [calibration_metrics[m]['MCE'] for m in models]
        brier_values = [calibration_metrics[m]['Brier'] for m in models]
        
        # ECE comparison
        axes[0].bar([m.upper() for m in models], ece_values, color='skyblue', alpha=0.7)
        axes[0].set_title('Expected Calibration Error (ECE)')
        axes[0].set_ylabel('ECE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MCE comparison
        axes[1].bar([m.upper() for m in models], mce_values, color='lightcoral', alpha=0.7)
        axes[1].set_title('Maximum Calibration Error (MCE)')
        axes[1].set_ylabel('MCE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Brier score comparison
        axes[2].bar([m.upper() for m in models], brier_values, color='lightgreen', alpha=0.7)
        axes[2].set_title('Brier Score')
        axes[2].set_ylabel('Brier Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        fig.savefig(os.path.join(VISUALS_DIR, "calibration_summary.svg"), 
                   format='svg', bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Save calibration metrics
    with open(os.path.join(REPORTS_DIR, "calibration_metrics.json"), 'w') as f:
        json.dump(calibration_metrics, f, indent=2)
    
    # Update thresholds with F1-optimal values
    updated_thresholds = {}
    for model_name in optimal_thresholds:
        # Use F1-optimal as the primary threshold
        updated_thresholds[model_name] = optimal_thresholds[model_name]['f1_optimal']
    
    # Save updated thresholds
    with open(os.path.join(REPORTS_DIR, "thresholds.json"), 'w') as f:
        json.dump(updated_thresholds, f, indent=2)
    
    # Save all threshold options
    with open(os.path.join(REPORTS_DIR, "threshold_options.json"), 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    print("\nCalibration Diagnostics Summary:")
    print("=" * 50)
    for model_name in calibration_metrics:
        metrics = calibration_metrics[model_name]
        thresholds = optimal_thresholds[model_name]
        print(f"\n{model_name.upper()}:")
        print(f"  ECE: {metrics['ECE']:.4f}")
        print(f"  MCE: {metrics['MCE']:.4f}")
        print(f"  Brier: {metrics['Brier']:.4f}")
        print(f"  F1-optimal threshold: {thresholds['f1_optimal']:.4f}")
        print(f"  85% recall threshold: {thresholds['recall_85']:.4f}")
        print(f"  Net benefit optimal: {thresholds['net_benefit_optimal']:.4f}")
    
    print(f"\nFiles saved:")
    print(f"  - Calibration metrics: {os.path.join(REPORTS_DIR, 'calibration_metrics.json')}")
    print(f"  - Updated thresholds: {os.path.join(REPORTS_DIR, 'thresholds.json')}")
    print(f"  - Threshold options: {os.path.join(REPORTS_DIR, 'threshold_options.json')}")
    print(f"  - Calibration curves: {VISUALS_DIR}/calibration_curve_*.svg")
    print(f"  - Decision curves: {os.path.join(VISUALS_DIR, 'decision_curves.svg')}")
    print(f"  - Summary plot: {os.path.join(VISUALS_DIR, 'calibration_summary.svg')}")


if __name__ == "__main__":
    main()