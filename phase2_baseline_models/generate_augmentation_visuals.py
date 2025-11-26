#!/usr/bin/env python3
"""
Phase 2: Data Augmentation Analysis and Visuals
Compares baseline (no augmentation) vs multiple augmentation techniques
Produces before/after visuals and performance comparisons
Includes clinical sanity checks to ensure augmentations remain plausible

Outputs:
visuals/augmentation/
 - class_distribution_comparison.png
 - roc_curves_comparison.png
 - auc_by_technique.png
 - metrics_bar_death_class.png
 - feature_shift_checks.png
reports/augmentation/
 - augmentation_summary.md
 - metrics_by_technique.json
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
   classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.metrics import average_precision_score, precision_recall_curve, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler

RANDOM_STATE = 42
DATA_PATH = '../clean_data.csv'
TARGET_COL = 'Death outcome (YES/NO)'

# Key numeric features for shift checks
KEY_NUMERIC = ['Age', 'BMI', 'Blood glucose at admission ', 'BP_sys', 'BP_dia']
# Add constants for missing-value augmentation focus
HBA1C_COL = 'HbA1c (last one before admission)'
HBA1C_AUG_K = 3  # number of synthetic draws per missing train row
MISSINGNESS_THRESHOLD = 0.0  # add indicators for any feature with missingness > 0

os.makedirs('visuals/augmentation', exist_ok=True)
os.makedirs('reports/augmentation', exist_ok=True)

print('=== Phase 2 Augmentation Analysis ===')
print('Loading data...')
df = pd.read_csv(DATA_PATH)
print(f'Dataset: {df.shape}')

if TARGET_COL not in df.columns:
   raise ValueError(f"Target column not found: {TARGET_COL}")

# Clean target: keep only valid yes/no rows
valid_mask = df[TARGET_COL].astype(str).str.lower().isin(['yes', 'no'])
invalid_count = (~valid_mask).sum()
if invalid_count > 0:
   print(f"Dropping {invalid_count} rows with invalid/missing target labels")

df = df.loc[valid_mask].reset_index(drop=True)

# Report HbA1c missingness early
if HBA1C_COL in df.columns:
   hba1c_missing_rate = df[HBA1C_COL].isna().mean() * 100
   print(f"HbA1c missing rate: {hba1c_missing_rate:.1f}%")
else:
   print("HbA1c column not found; skipping missing-value augmentation for it.")

# Drop obvious IDs and operational timing columns
exclude_cols = ['CMRN', 'Unnamed: 0', 'Hospital admission timing',
               'Fibrinolytic therapy timing', 'Timing of CT scan']
feature_cols = [c for c in df.columns if c not in exclude_cols + [TARGET_COL]]
X_raw = df[feature_cols].copy()
y_raw = df[TARGET_COL].astype(str).str.lower().map({'no': 0, 'yes': 1})

# Encode categoricals and impute
le = LabelEncoder()
X_enc = X_raw.copy()
for c in X_enc.columns:
    if X_enc[c].dtype == 'object':
        X_enc[c] = X_enc[c].fillna('unknown').astype(str)
        X_enc[c] = le.fit_transform(X_enc[c])
# Add HbA1c missingness indicator as a feature (before imputation) if available
if HBA1C_COL in X_raw.columns:
   X_enc['HbA1c_missing'] = X_raw[HBA1C_COL].isna().astype(int)
# Add missingness indicators for all columns (before imputation)
missing_rates = X_raw.isna().mean()
for col, rate in missing_rates.items():
   if rate > MISSINGNESS_THRESHOLD:
       X_enc[f'{col}_missing'] = X_raw[col].isna().astype(int)

imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X_enc), columns=X_enc.columns)

# Keep a raw split to detect missingness locations for augmentation
X_train_raw, X_test_raw, _, _ = train_test_split(
   X_raw, y_raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y_raw
)

X_train, X_test, y_train, y_test = train_test_split(
   X_imp, y_raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y_raw
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f'Train: {len(y_train)} | Test: {len(y_test)}')

# Baseline model (consistent with Phase 2)
BASE_MODEL = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)

# Define augmentation techniques
techniques = {
   'none': None,
   'random_over': RandomOverSampler(random_state=RANDOM_STATE),
   'smote': SMOTE(random_state=RANDOM_STATE),
   'bsmote': BorderlineSMOTE(random_state=RANDOM_STATE),
   'adasyn': ADASYN(random_state=RANDOM_STATE),
}

results = {}
roc_curves = {}
class_dist = {}

# Helper: train/eval
def train_eval(Xtr, ytr, Xte, yte, label, calibrated=False):
   clf = BASE_MODEL
   if calibrated:
       clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
   clf.fit(Xtr, ytr)
   y_prob = clf.predict_proba(Xte)[:, 1]
   y_pred = (y_prob >= 0.5).astype(int)
   auc = roc_auc_score(yte, y_prob)
   auprc = average_precision_score(yte, y_prob)
   brier = brier_score_loss(yte, y_prob)
   report = classification_report(yte, y_pred, output_dict=True)
   results[label] = {
       'auc': float(auc),
       'auprc': float(auprc),
       'brier': float(brier),
       'accuracy': float(report['accuracy']),
       'precision_1': float(report['1']['precision']),
       'recall_1': float(report['1']['recall']),
       'f1_1': float(report['1']['f1-score'])
   }
   fpr, tpr, _ = roc_curve(yte, y_prob)
   roc_curves[label] = (fpr, tpr)
   # store PR curve and probs for threshold sweep
   pr = precision_recall_curve(yte, y_prob)
   results[label]['pr_curve'] = {'precision': pr[0].tolist(), 'recall': pr[1].tolist(), 'thresholds': pr[2].tolist() if len(pr) > 2 else []}
   results[label]['probs'] = y_prob.tolist()

# Helper: get class distribution
def class_distribution(y, name):
   vc = pd.Series(y).value_counts().to_dict()
   class_dist[name] = {0: int(vc.get(0, 0)), 1: int(vc.get(1, 0))}

# Baseline (no augmentation)
class_distribution(y_train, 'none_train')
train_eval(X_train_s, y_train, X_test_s, y_test, 'none')

# Add: class_weight baseline (no augmentation)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier as RFC
cls_weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=y_train)
BASE_MODEL_WEIGHTED = RFC(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, class_weight={0: cls_weights[0], 1: cls_weights[1]})
# Evaluate weighted model
_tmp = BASE_MODEL
BASE_MODEL = BASE_MODEL_WEIGHTED
train_eval(X_train_s, y_train, X_test_s, y_test, 'class_weight')
BASE_MODEL = _tmp

# Add: calibrated baseline (isotonic)
train_eval(X_train_s, y_train, X_test_s, y_test, 'none_calibrated', calibrated=True)

# HbA1c-focused missing-value augmentation (train only)
def augment_hba1c_train(X_train_unscaled: pd.DataFrame, X_train_raw_unscaled: pd.DataFrame, y_train_ser: pd.Series, k: int = HBA1C_AUG_K):
   if HBA1C_COL not in X_train_unscaled.columns or HBA1C_COL not in X_train_raw_unscaled.columns:
       return X_train_unscaled.copy(), y_train_ser.copy()
   miss_mask = X_train_raw_unscaled[HBA1C_COL].isna()
   if miss_mask.sum() == 0:
       return X_train_unscaled.copy(), y_train_ser.copy()

   rows = []
   ys = []
   # Helper to decide high vs low HbA1c distribution
   def is_high_risk_row(idx):
       high = False
       # Diabetes flag
       col_diab = 'known case of diabetes (YES/NO)'
       if col_diab in X_train_raw_unscaled.columns:
           try:
               val = str(X_train_raw_unscaled.loc[idx, col_diab]).strip().lower()
               if val == 'yes':
                   high = True
           except Exception:
               pass
       # Elevated admission glucose as proxy
       if (not high) and ('Blood glucose at admission ' in X_train_raw_unscaled.columns):
           try:
               bg = float(X_train_raw_unscaled.loc[idx, 'Blood glucose at admission '])
               if bg >= 180:
                   high = True
           except Exception:
               pass
       return high

   for idx in X_train_unscaled.index[miss_mask]:
       base_row = X_train_unscaled.loc[idx].copy()
       high = is_high_risk_row(idx)
       mu, sigma = (8.5, 1.2) if high else (5.7, 0.5)
       for _ in range(k):
           new_row = base_row.copy()
           draw = float(np.clip(np.random.normal(mu, sigma), 4.0, 14.0))
           new_row[HBA1C_COL] = draw
           rows.append(new_row)
           ys.append(y_train_ser.loc[idx])

   if rows:
       X_gen = pd.DataFrame(rows, columns=X_train_unscaled.columns)
       y_gen = pd.Series(ys, name=y_train_ser.name)
       X_aug = pd.concat([X_train_unscaled, X_gen], axis=0).reset_index(drop=True)
       y_aug = pd.concat([y_train_ser, y_gen], axis=0).reset_index(drop=True)
       return X_aug, y_aug
   else:
       return X_train_unscaled.copy(), y_train_ser.copy()

# Apply HbA1c augmentation and evaluate as its own technique
if HBA1C_COL in X_train.columns:
   X_hb_aug_unscaled, y_hb_aug = augment_hba1c_train(X_train, X_train_raw, y_train, k=HBA1C_AUG_K)
   X_hb_aug_scaled = scaler.transform(X_hb_aug_unscaled)
   class_distribution(y_hb_aug, 'hba1c_aug_train')
   train_eval(X_hb_aug_scaled, y_hb_aug, X_test_s, y_test, 'hba1c_aug')

# Multi-Imputation augmentation (IterativeImputer) on train only
def multi_impute_augment_train(X_enc_train: pd.DataFrame, y_train_ser: pd.Series, seeds=(42,43,44,45,46), max_iter: int = 10):
   imputations = []
   for seed in seeds:
       imp = IterativeImputer(random_state=seed, sample_posterior=True, max_iter=max_iter, initial_strategy='median')
       Xi = pd.DataFrame(imp.fit_transform(X_enc_train), columns=X_enc_train.columns, index=X_enc_train.index)
       imputations.append(Xi)
   X_mi_aug = pd.concat(imputations, axis=0)
   y_mi_aug = pd.concat([y_train_ser] * len(seeds), axis=0)
   return X_mi_aug, y_mi_aug

# Prepare train view with original (pre-impute) encodings to allow MI
X_enc_train = X_enc.loc[X_train.index]
X_mi_aug_unscaled, y_mi_aug = multi_impute_augment_train(X_enc_train, y_train)
X_mi_aug_scaled = scaler.transform(X_mi_aug_unscaled)
class_distribution(y_mi_aug, 'multi_impute_aug_train')
train_eval(X_mi_aug_scaled, y_mi_aug, X_test_s, y_test, 'multi_impute_aug')

# Apply augmentations
for name, augmenter in techniques.items():
    if name == 'none':
        continue
    print(f'Applying augmentation: {name}')
    X_aug, y_aug = augmenter.fit_resample(X_train_s, y_train)
    class_distribution(y_aug, f'{name}_train')
    train_eval(X_aug, y_aug, X_test_s, y_test, name)

# Clinical sanity checks on augmented data vs original (train set only)
# Compare distributions of key numeric features (where available)
available_keys = [c for c in KEY_NUMERIC if c in X_imp.columns]

fig, axes = plt.subplots(2, len(available_keys), figsize=(5*len(available_keys), 8))
fig.suptitle('Feature Shift Checks (Train Set) - Original vs Augmented', fontsize=14, fontweight='bold')

# Plot original distributions
for j, col in enumerate(available_keys):
    sns.kdeplot(x=X_train[col], ax=axes[0, j], label='Original', color='blue')
    axes[0, j].set_title(f'{col} (Original)')
    axes[0, j].grid(True, alpha=0.3)

# For one augmentation (SMOTE) to visualize shift
if 'smote_train' in class_dist:
    # We need the augmented samples in feature space for plotting; rebuild SMOTE on unscaled
    smote = SMOTE(random_state=RANDOM_STATE)
    Xtr_unscaled = X_train.copy()
    # Refit scaler to unscaled for consistency with modeling (not used here)
    X_sm, y_sm = smote.fit_resample(Xtr_unscaled, y_train)
    for j, col in enumerate(available_keys):
        sns.kdeplot(x=X_sm[col], ax=axes[1, j], label='SMOTE', color='red')
        axes[1, j].set_title(f'{col} (After SMOTE)')
        axes[1, j].grid(True, alpha=0.3)
else:
    for j in range(len(available_keys)):
        axes[1, j].axis('off')

plt.tight_layout()
plt.savefig('visuals/augmentation/feature_shift_checks.png', dpi=300, bbox_inches='tight')
plt.close()

# 1) Class distribution comparison (before/after for techniques)
# Build labels dynamically to include HbA1c augmentation when present
labels = ['none'] + [t for t in ['hba1c_aug', 'random_over', 'smote', 'bsmote', 'adasyn'] if f'{t}_train' in class_dist]
labels = ['none'] + [t for t in ['hba1c_aug', 'multi_impute_aug', 'random_over', 'smote', 'bsmote', 'adasyn'] if f'{t}_train' in class_dist]
orig_counts = [class_dist.get('none_train', {}).get(0, 0), class_dist.get('none_train', {}).get(1, 0)]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# Original
ax[0].bar(['Alive(0)', 'Death(1)'], orig_counts, color=['lightgreen', 'salmon'])
ax[0].set_title('Original Train Class Distribution')
ax[0].set_ylabel('Count')
# After per technique
tech_counts = []
for t in labels[1:]:
    c = class_dist.get(f'{t}_train', {0: 0, 1: 0})
    tech_counts.append([c[0], c[1]])
tech_counts = np.array(tech_counts) if tech_counts else np.zeros((0, 2))
ax[1].bar(np.arange(len(labels)-1)-0.15, tech_counts[:,0] if len(tech_counts)>0 else [], width=0.3, label='Alive(0)', color='lightgreen')
ax[1].bar(np.arange(len(labels)-1)+0.15, tech_counts[:,1] if len(tech_counts)>0 else [], width=0.3, label='Death(1)', color='salmon')
ax[1].set_xticks(np.arange(len(labels)-1))
ax[1].set_xticklabels(labels[1:])
ax[1].set_title('After Augmentation (Train)')
ax[1].legend()
plt.tight_layout()
plt.savefig('visuals/augmentation/class_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2) ROC curves comparison
plt.figure(figsize=(8, 6))
for name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{name.upper()} (AUC={results[name]["auc"]:.3f})')
plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Baseline vs Augmentations (RF)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/augmentation/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Add: PR curves comparison and AUPRC bar
plt.figure(figsize=(8, 6))
for name in results.keys():
    pr = results[name].get('pr_curve')
    if pr:
        plt.plot(pr['recall'], pr['precision'], label=f"{name.upper()} (AUPRC={results[name]['auprc']:.3f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves by Technique')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/augmentation/pr_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# AUPRC bar chart
order_pr = list(results.keys())
auprcs = [results[k]['auprc'] for k in order_pr]
plt.figure(figsize=(8, 5))
colors = ['steelblue' if k=='none' else 'indianred' for k in order_pr]
plt.bar(order_pr, auprcs, color=colors)
for i, v in enumerate(auprcs):
    plt.text(i, v+0.01, f'{v:.3f}', ha='center')
plt.ylim(0, 1)
plt.title('AUPRC by Technique (Higher is Better)')
plt.ylabel('AUPRC')
plt.tight_layout()
plt.savefig('visuals/augmentation/auprc_by_technique.png', dpi=300, bbox_inches='tight')
plt.close()

# Add: Threshold sweep for F1 on positive class
thr = np.linspace(0.05, 0.95, 19)
plt.figure(figsize=(10, 6))
for name in ['none', 'class_weight', 'smote', 'bsmote', 'adasyn', 'hba1c_aug', 'multi_impute_aug']:
    if name in results and 'probs' in results[name]:
        y_prob = np.array(results[name]['probs'])
        f1s = []
        for t in thr:
            y_pred = (y_prob >= t).astype(int)
            try:
                from sklearn.metrics import f1_score
                f1s.append(f1_score(y_test, y_pred))
            except Exception:
                f1s.append(np.nan)
        plt.plot(thr, f1s, label=name)
plt.xlabel('Decision Threshold')
plt.ylabel('F1 (positive class)')
plt.title('Threshold Sweep (F1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/augmentation/threshold_sweep_f1.png', dpi=300, bbox_inches='tight')
plt.close()

# Add: Calibration curve for selected models
plt.figure(figsize=(8, 6))
for name in ['none', 'none_calibrated', 'class_weight']:
    if name in results and 'probs' in results[name]:
        y_prob = np.array(results[name]['probs'])
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
        plt.plot(mean_pred, frac_pos, marker='o', label=name)
plt.plot([0,1],[0,1],'k--', alpha=0.6)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Reliability)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/augmentation/calibration_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 3) AUC by technique
order = list(results.keys())
order = list(results.keys())
aucs = [results[k]['auc'] for k in order]
plt.figure(figsize=(8, 5))
colors = ['steelblue' if k=='none' else 'indianred' for k in order]
plt.bar(order, aucs, color=colors)
for i, v in enumerate(aucs):
    plt.text(i, v+0.01, f'{v:.3f}', ha='center')
plt.ylim(0, 1)
plt.title('AUC by Technique (Higher is Better)')
plt.ylabel('AUC')
plt.tight_layout()
plt.savefig('visuals/augmentation/auc_by_technique.png', dpi=300, bbox_inches='tight')
plt.close()

# 4) Metrics bar (Death class)
metrics = ['precision_1', 'recall_1', 'f1_1']
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, m in enumerate(metrics):
    vals = [results[k][m] for k in order]
    ax[i].bar(order, vals, color=colors)
    ax[i].set_title(m.replace('_', ' ').title())
    ax[i].set_ylim(0, 1)
    for j, v in enumerate(vals):
        ax[i].text(j, v+0.01, f'{v:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('visuals/augmentation/metrics_bar_death_class.png', dpi=300, bbox_inches='tight')
plt.close()

# 5) Confusion matrices for baseline vs SMOTE (if available)
from sklearn.metrics import confusion_matrix
if 'none' in results:
    # Recompute predictions for plots
    base_clf = BASE_MODEL
    base_clf.fit(X_train_s, y_train)
    y_pred_none = base_clf.predict(X_test_s)
    cm_none = confusion_matrix(y_test, y_pred_none)
else:
    cm_none = None

cm_smote = None
if 'smote' in results:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_sm_tr, y_sm_tr = sm.fit_resample(X_train_s, y_train)
    sm_clf = BASE_MODEL
    sm_clf.fit(X_sm_tr, y_sm_tr)
    y_pred_sm = sm_clf.predict(X_test_s)
    cm_smote = confusion_matrix(y_test, y_pred_sm)

if cm_none is not None or cm_smote is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if cm_none is not None:
        sns.heatmap(cm_none, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix: Baseline (None)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
    else:
        axes[0].axis('off')
    if cm_smote is not None:
        sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
        axes[1].set_title('Confusion Matrix: SMOTE')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
    else:
        axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('visuals/augmentation/confusion_matrices_none_vs_smote.png', dpi=300, bbox_inches='tight')
    plt.close()

# Save metrics
import json
with open('reports/augmentation/metrics_by_technique.json', 'w') as f:
    json.dump(results, f, indent=2)

# Clinical plausibility checks & summary
summary_lines = []
summary_lines.append('# Phase 2 Augmentation Summary')
summary_lines.append('')
summary_lines.append('Techniques evaluated: none, random_over, SMOTE, BorderlineSMOTE, ADASYN, HbA1c-missing augmentation (train-only), Multi-Imputation augmentation (train-only)')
summary_lines.append('')
# Death rate baseline
death_rate = (y_raw.sum() / len(y_raw)) * 100
summary_lines.append(f'- Original death rate: {death_rate:.1f}% (class imbalance)')
# Report HbA1c missingness and augmentation details
if HBA1C_COL in df.columns:
    summary_lines.append(f'- HbA1c missing rate: {df[HBA1C_COL].isna().mean()*100:.1f}% in original dataset')
    if 'hba1c_aug' in results:
        try:
            miss_train_cnt = int(X_train_raw[HBA1C_COL].isna().sum())
            summary_lines.append(f'- HbA1c augmentation: created {miss_train_cnt * HBA1C_AUG_K} synthetic train rows ({HBA1C_AUG_K} draws per missing row) with clinically plausible values conditioned on diabetes/glucose status.')
        except Exception:
            pass
if 'multi_impute_aug' in results:
   summary_lines.append('- Multi-Imputation augmentation: stacked multiple IterativeImputer draws (posterior sampling) on the training set; test set untouched. See missingness visuals for coverage and distribution checks.')
# Check if any augmentation overly distorts AUC or metrics
best_auc_name = max(results, key=lambda k: results[k]['auc'])
summary_lines.append(f'- Best AUC: {results[best_auc_name]["auc"]:.3f} with {best_auc_name}')
# Clinical sanity: compare medians for key features before vs after SMOTE
if 'smote_train' in class_dist:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    shifts = []
    for col in available_keys:
        med_orig = float(np.nanmedian(X_train[col]))
        med_sm = float(np.nanmedian(X_sm[col]))
        shift = med_sm - med_orig
        shifts.append((col, med_orig, med_sm, shift))
    summary_lines.append('\n## Feature Median Shifts (SMOTE)')
    for col, m0, m1, d in shifts:
        summary_lines.append(f'- {col}: {m0:.2f} -> {m1:.2f} (Δ {d:+.2f})')
    large_shifts = [s for s in shifts if abs(s[3]) > (0.25 * (np.nanstd(X_train[s[0]]) + 1e-6))]
    if large_shifts:
        summary_lines.append('\nWarning: Some features show notable median shifts after SMOTE. Verify clinical sense.')
else:
    summary_lines.append('\nSMOTE results unavailable for feature shift checks.')

summary_lines.append('\n## Interpretation Guidance')
summary_lines.append('- Improvements in recall may come at the cost of precision. Focus on F1 and AUC, not accuracy alone.')
summary_lines.append('- If augmentation substantially distorts key clinical feature distributions, treat results with caution.')

with open('reports/augmentation/augmentation_summary.md', 'w') as f:
    f.write('\n'.join(summary_lines))

print('Saved visuals to visuals/augmentation and report to reports/augmentation')
print('=== Phase 2 Augmentation Analysis Complete ===')

# Missingness visualizations: rates and heatmap
# 5a) Missingness rates bar (top 25)
miss_rates = df[feature_cols].isna().mean().sort_values(ascending=False)
plt.figure(figsize=(10, min(12, 0.4*len(miss_rates.head(25)) + 2)))
miss_rates.head(25).plot(kind='barh', color='slateblue')
plt.gca().invert_yaxis()
plt.xlabel('Missing Rate')
plt.title('Top Feature Missingness Rates')
plt.tight_layout()
plt.savefig('visuals/augmentation/missingness_rates.png', dpi=300, bbox_inches='tight')
plt.close()

# 5b) Missingness heatmap for most-missing 30 columns (sample of rows for readability)
top_cols = miss_rates.head(30).index.tolist()
mask = df[top_cols].isna()
if len(df) > 200:
   mask = mask.sample(n=200, random_state=RANDOM_STATE)
plt.figure(figsize=(12, 8))
sns.heatmap(mask.T, cbar=False, cmap=[[1,1,1],[0.2,0.2,0.2]], linewidths=0.1)
plt.ylabel('Features')
plt.xlabel('Sampled Rows')
plt.title('Missingness Heatmap (1=Missing)')
plt.tight_layout()
plt.savefig('visuals/augmentation/missingness_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5c) Imputation distribution overlays for top-k numeric features
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
top_numeric = [c for c in miss_rates.index if c in numeric_cols][:6]
if len(top_numeric) > 0:
   rows = int(np.ceil(len(top_numeric)/3))
   fig, axes = plt.subplots(rows, 3, figsize=(15, 4*rows))
   axes = axes.flatten()
   # Use first MI draw to visualize imputed values
   imp_vis = IterativeImputer(random_state=RANDOM_STATE, sample_posterior=True, max_iter=10, initial_strategy='median')
   Xi_vis = pd.DataFrame(imp_vis.fit_transform(X_enc_train), columns=X_enc_train.columns, index=X_enc_train.index)
   for i, col in enumerate(top_numeric):
       ax = axes[i]
       orig_series = X_enc_train[col]
       miss_mask_train = orig_series.isna()
       observed = orig_series[~miss_mask_train].astype(float)
       imputed = Xi_vis.loc[miss_mask_train, col].astype(float)
       if len(observed) > 0:
           sns.kdeplot(x=observed, ax=ax, color='blue', label='Observed')
       if len(imputed) > 0:
           sns.kdeplot(x=imputed, ax=ax, color='orange', label='Imputed')
       ax.set_title(f'{col} (Obs vs Imputed)')
       ax.legend()
       ax.grid(True, alpha=0.3)
   for j in range(i+1, len(axes)):
       axes[j].axis('off')
   plt.tight_layout()
   plt.savefig('visuals/augmentation/imputation_distributions.png', dpi=300, bbox_inches='tight')
   plt.close()

# 1) Class distribution comparison (before/after for techniques)
# Build labels dynamically to include HbA1c augmentation when present
labels = ['none'] + [t for t in ['hba1c_aug', 'random_over', 'smote', 'bsmote', 'adasyn'] if f'{t}_train' in class_dist]
labels = ['none'] + [t for t in ['hba1c_aug', 'multi_impute_aug', 'random_over', 'smote', 'bsmote', 'adasyn'] if f'{t}_train' in class_dist]
orig_counts = [class_dist.get('none_train', {}).get(0, 0), class_dist.get('none_train', {}).get(1, 0)]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# Original
ax[0].bar(['Alive(0)', 'Death(1)'], orig_counts, color=['lightgreen', 'salmon'])
ax[0].set_title('Original Train Class Distribution')
ax[0].set_ylabel('Count')
# After per technique
tech_counts = []
for t in labels[1:]:
    c = class_dist.get(f'{t}_train', {0: 0, 1: 0})
    tech_counts.append([c[0], c[1]])
tech_counts = np.array(tech_counts) if tech_counts else np.zeros((0, 2))
ax[1].bar(np.arange(len(labels)-1)-0.15, tech_counts[:,0] if len(tech_counts)>0 else [], width=0.3, label='Alive(0)', color='lightgreen')
ax[1].bar(np.arange(len(labels)-1)+0.15, tech_counts[:,1] if len(tech_counts)>0 else [], width=0.3, label='Death(1)', color='salmon')
ax[1].set_xticks(np.arange(len(labels)-1))
ax[1].set_xticklabels(labels[1:])
ax[1].set_title('After Augmentation (Train)')
ax[1].legend()
plt.tight_layout()
plt.savefig('visuals/augmentation/class_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2) ROC curves comparison
plt.figure(figsize=(8, 6))
for name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{name.upper()} (AUC={results[name]["auc"]:.3f})')
plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Baseline vs Augmentations (RF)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/augmentation/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Add: PR curves comparison and AUPRC bar
plt.figure(figsize=(8, 6))
for name in results.keys():
    pr = results[name].get('pr_curve')
    if pr:
        plt.plot(pr['recall'], pr['precision'], label=f"{name.upper()} (AUPRC={results[name]['auprc']:.3f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves by Technique')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/augmentation/pr_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# AUPRC bar chart
order_pr = list(results.keys())
auprcs = [results[k]['auprc'] for k in order_pr]
plt.figure(figsize=(8, 5))
colors = ['steelblue' if k=='none' else 'indianred' for k in order_pr]
plt.bar(order_pr, auprcs, color=colors)
for i, v in enumerate(auprcs):
    plt.text(i, v+0.01, f'{v:.3f}', ha='center')
plt.ylim(0, 1)
plt.title('AUPRC by Technique (Higher is Better)')
plt.ylabel('AUPRC')
plt.tight_layout()
plt.savefig('visuals/augmentation/auprc_by_technique.png', dpi=300, bbox_inches='tight')
plt.close()

# Add: Threshold sweep for F1 on positive class
thr = np.linspace(0.05, 0.95, 19)
plt.figure(figsize=(10, 6))
for name in ['none', 'class_weight', 'smote', 'bsmote', 'adasyn', 'hba1c_aug', 'multi_impute_aug']:
    if name in results and 'probs' in results[name]:
        y_prob = np.array(results[name]['probs'])
        f1s = []
        for t in thr:
            y_pred = (y_prob >= t).astype(int)
            try:
                from sklearn.metrics import f1_score
                f1s.append(f1_score(y_test, y_pred))
            except Exception:
                f1s.append(np.nan)
        plt.plot(thr, f1s, label=name)
plt.xlabel('Decision Threshold')
plt.ylabel('F1 (positive class)')
plt.title('Threshold Sweep (F1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/augmentation/threshold_sweep_f1.png', dpi=300, bbox_inches='tight')
plt.close()

# Add: Calibration curve for selected models
plt.figure(figsize=(8, 6))
for name in ['none', 'none_calibrated', 'class_weight']:
    if name in results and 'probs' in results[name]:
        y_prob = np.array(results[name]['probs'])
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
        plt.plot(mean_pred, frac_pos, marker='o', label=name)
plt.plot([0,1],[0,1],'k--', alpha=0.6)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Reliability)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/augmentation/calibration_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 3) AUC by technique
order = list(results.keys())
order = list(results.keys())
aucs = [results[k]['auc'] for k in order]
plt.figure(figsize=(8, 5))
colors = ['steelblue' if k=='none' else 'indianred' for k in order]
plt.bar(order, aucs, color=colors)
for i, v in enumerate(aucs):
    plt.text(i, v+0.01, f'{v:.3f}', ha='center')
plt.ylim(0, 1)
plt.title('AUC by Technique (Higher is Better)')
plt.ylabel('AUC')
plt.tight_layout()
plt.savefig('visuals/augmentation/auc_by_technique.png', dpi=300, bbox_inches='tight')
plt.close()

# 4) Metrics bar (Death class)
metrics = ['precision_1', 'recall_1', 'f1_1']
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, m in enumerate(metrics):
    vals = [results[k][m] for k in order]
    ax[i].bar(order, vals, color=colors)
    ax[i].set_title(m.replace('_', ' ').title())
    ax[i].set_ylim(0, 1)
    for j, v in enumerate(vals):
        ax[i].text(j, v+0.01, f'{v:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('visuals/augmentation/metrics_bar_death_class.png', dpi=300, bbox_inches='tight')
plt.close()

# 5) Confusion matrices for baseline vs SMOTE (if available)
from sklearn.metrics import confusion_matrix
if 'none' in results:
    # Recompute predictions for plots
    base_clf = BASE_MODEL
    base_clf.fit(X_train_s, y_train)
    y_pred_none = base_clf.predict(X_test_s)
    cm_none = confusion_matrix(y_test, y_pred_none)
else:
    cm_none = None

cm_smote = None
if 'smote' in results:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_sm_tr, y_sm_tr = sm.fit_resample(X_train_s, y_train)
    sm_clf = BASE_MODEL
    sm_clf.fit(X_sm_tr, y_sm_tr)
    y_pred_sm = sm_clf.predict(X_test_s)
    cm_smote = confusion_matrix(y_test, y_pred_sm)

if cm_none is not None or cm_smote is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if cm_none is not None:
        sns.heatmap(cm_none, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix: Baseline (None)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
    else:
        axes[0].axis('off')
    if cm_smote is not None:
        sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
        axes[1].set_title('Confusion Matrix: SMOTE')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
    else:
        axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('visuals/augmentation/confusion_matrices_none_vs_smote.png', dpi=300, bbox_inches='tight')
    plt.close()

# Save metrics
import json
with open('reports/augmentation/metrics_by_technique.json', 'w') as f:
    json.dump(results, f, indent=2)

# Clinical plausibility checks & summary
summary_lines = []
summary_lines.append('# Phase 2 Augmentation Summary')
summary_lines.append('')
summary_lines.append('Techniques evaluated: none, random_over, SMOTE, BorderlineSMOTE, ADASYN, HbA1c-missing augmentation (train-only), Multi-Imputation augmentation (train-only)')
summary_lines.append('')
# Death rate baseline
death_rate = (y_raw.sum() / len(y_raw)) * 100
summary_lines.append(f'- Original death rate: {death_rate:.1f}% (class imbalance)')
# Report HbA1c missingness and augmentation details
if HBA1C_COL in df.columns:
    summary_lines.append(f'- HbA1c missing rate: {df[HBA1C_COL].isna().mean()*100:.1f}% in original dataset')
    if 'hba1c_aug' in results:
        try:
            miss_train_cnt = int(X_train_raw[HBA1C_COL].isna().sum())
            summary_lines.append(f'- HbA1c augmentation: created {miss_train_cnt * HBA1C_AUG_K} synthetic train rows ({HBA1C_AUG_K} draws per missing row) with clinically plausible values conditioned on diabetes/glucose status.')
        except Exception:
            pass
if 'multi_impute_aug' in results:
   summary_lines.append('- Multi-Imputation augmentation: stacked multiple IterativeImputer draws (posterior sampling) on the training set; test set untouched. See missingness visuals for coverage and distribution checks.')
# Check if any augmentation overly distorts AUC or metrics
best_auc_name = max(results, key=lambda k: results[k]['auc'])
summary_lines.append(f'- Best AUC: {results[best_auc_name]["auc"]:.3f} with {best_auc_name}')
# Clinical sanity: compare medians for key features before vs after SMOTE
if 'smote_train' in class_dist:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    shifts = []
    for col in available_keys:
        med_orig = float(np.nanmedian(X_train[col]))
        med_sm = float(np.nanmedian(X_sm[col]))
        shift = med_sm - med_orig
        shifts.append((col, med_orig, med_sm, shift))
    summary_lines.append('\n## Feature Median Shifts (SMOTE)')
    for col, m0, m1, d in shifts:
        summary_lines.append(f'- {col}: {m0:.2f} -> {m1:.2f} (Δ {d:+.2f})')
    large_shifts = [s for s in shifts if abs(s[3]) > (0.25 * (np.nanstd(X_train[s[0]]) + 1e-6))]
    if large_shifts:
        summary_lines.append('\nWarning: Some features show notable median shifts after SMOTE. Verify clinical sense.')
else:
    summary_lines.append('\nSMOTE results unavailable for feature shift checks.')

summary_lines.append('\n## Interpretation Guidance')
summary_lines.append('- Improvements in recall may come at the cost of precision. Focus on F1 and AUC, not accuracy alone.')
summary_lines.append('- If augmentation substantially distorts key clinical feature distributions, treat results with caution.')

with open('reports/augmentation/augmentation_summary.md', 'w') as f:
    f.write('\n'.join(summary_lines))

print('Saved visuals to visuals/augmentation and report to reports/augmentation')
print('=== Phase 2 Augmentation Analysis Complete ===')

# Missingness visualizations: rates and heatmap
# 5a) Missingness rates bar (top 25)
miss_rates = df[feature_cols].isna().mean().sort_values(ascending=False)
plt.figure(figsize=(10, min(12, 0.4*len(miss_rates.head(25)) + 2)))
miss_rates.head(25).plot(kind='barh', color='slateblue')
plt.gca().invert_yaxis()
plt.xlabel('Missing Rate')
plt.title('Top Feature Missingness Rates')
plt.tight_layout()
plt.savefig('visuals/augmentation/missingness_rates.png', dpi=300, bbox_inches='tight')
plt.close()

# 5b) Missingness heatmap for most-missing 30 columns (sample of rows for readability)
top_cols = miss_rates.head(30).index.tolist()
mask = df[top_cols].isna()
if len(df) > 200:
   mask = mask.sample(n=200, random_state=RANDOM_STATE)
plt.figure(figsize=(12, 8))
sns.heatmap(mask.T, cbar=False, cmap=[[1,1,1],[0.2,0.2,0.2]], linewidths=0.1)
plt.ylabel('Features')
plt.xlabel('Sampled Rows')
plt.title('Missingness Heatmap (1=Missing)')
plt.tight_layout()
plt.savefig('visuals/augmentation/missingness_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5c) Imputation distribution overlays for top-k numeric features
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
top_numeric = [c for c in miss_rates.index if c in numeric_cols][:6]
if len(top_numeric) > 0:
   rows = int(np.ceil(len(top_numeric)/3))
   fig, axes = plt.subplots(rows, 3, figsize=(15, 4*rows))
   axes = axes.flatten()
   # Use first MI draw to visualize imputed values
   imp_vis = IterativeImputer(random_state=RANDOM_STATE, sample_posterior=True, max_iter=10, initial_strategy='median')
   Xi_vis = pd.DataFrame(imp_vis.fit_transform(X_enc_train), columns=X_enc_train.columns, index=X_enc_train.index)
   for i, col in enumerate(top_numeric):
       ax = axes[i]
       orig_series = X_enc_train[col]
       miss_mask_train = orig_series.isna()
       observed = orig_series[~miss_mask_train].astype(float)
       imputed = Xi_vis.loc[miss_mask_train, col].astype(float)
       if len(observed) > 0:
           sns.kdeplot(x=observed, ax=ax, color='blue', label='Observed')
       if len(imputed) > 0:
           sns.kdeplot(x=imputed, ax=ax, color='orange', label='Imputed')
       ax.set_title(f'{col} (Obs vs Imputed)')
       ax.legend()
       ax.grid(True, alpha=0.3)
   for j in range(i+1, len(axes)):
       axes[j].axis('off')
   plt.tight_layout()
   plt.savefig('visuals/augmentation/imputation_distributions.png', dpi=300, bbox_inches='tight')
   plt.close()

# 1) Class distribution comparison (before/after for techniques)
# Build labels dynamically to include HbA1c augmentation when present
labels = ['none'] + [t for t in ['hba1c_aug', 'random_over', 'smote', 'bsmote', 'adasyn'] if f'{t}_train' in class_dist]
labels = ['none'] + [t for t in ['hba1c_aug', 'multi_impute_aug', 'random_over', 'smote', 'bsmote', 'adasyn'] if f'{t}_train' in class_dist]
orig_counts = [class_dist.get('none_train', {}).get(0, 0), class_dist.get('none_train', {}).get(1, 0)]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# Original
ax[0].bar(['Alive(0)', 'Death(1)'], orig_counts, color=['lightgreen', 'salmon'])
ax[0].set_title('Original Train Class Distribution')
ax[0].set_ylabel('Count')
# After per technique
tech_counts = []
for t in labels[1:]:
    c = class_dist.get(f'{t}_train', {0: 0, 1: 0})
    tech_counts.append([c[0], c[1]])
tech_counts = np.array(tech_counts) if tech_counts else np.zeros((0, 2))
ax[1].bar(np.arange(len(labels)-1)-0.15, tech_counts[:,0] if len(tech_counts)>0 else [], width=0.3, label='Alive(0)', color='lightgreen')
ax[1].bar(np.arange(len(labels)-1)+0.15, tech_counts[:,1] if len(tech_counts)>0 else [], width=0.3, label='Death(1)', color='salmon')
ax[1].set_xticks(np.arange(len(labels)-1))
ax[1].set_xticklabels(labels[1:])
ax[1].set_title('After Augmentation (Train)')
ax[1].legend()
plt.tight_layout()
plt.savefig('visuals/augmentation/class_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2) ROC curves comparison
plt.figure(figsize=(8, 6))
for name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{name.upper()} (AUC={results[name]["auc"]:.3f})')
plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Baseline vs Augmentations (RF)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/augmentation/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3) AUC by technique
order = list(results.keys())
order = list(results.keys())
aucs = [results[k]['auc'] for k in order]
plt.figure(figsize=(8, 5))
colors = ['steelblue' if k=='none' else 'indianred' for k in order]
plt.bar(order, aucs, color=colors)
for i, v in enumerate(aucs):
    plt.text(i, v+0.01, f'{v:.3f}', ha='center')
plt.ylim(0, 1)
plt.title('AUC by Technique (Higher is Better)')
plt.ylabel('AUC')
plt.tight_layout()
plt.savefig('visuals/augmentation/auc_by_technique.png', dpi=300, bbox_inches='tight')
plt.close()

# 4) Metrics bar (Death class)
metrics = ['precision_1', 'recall_1', 'f1_1']
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, m in enumerate(metrics):
    vals = [results[k][m] for k in order]
    ax[i].bar(order, vals, color=colors)
    ax[i].set_title(m.replace('_', ' ').title())
    ax[i].set_ylim(0, 1)
    for j, v in enumerate(vals):
        ax[i].text(j, v+0.01, f'{v:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('visuals/augmentation/metrics_bar_death_class.png', dpi=300, bbox_inches='tight')
plt.close()

# 5) Confusion matrices for baseline vs SMOTE (if available)
from sklearn.metrics import confusion_matrix
if 'none' in results:
    # Recompute predictions for plots
    base_clf = BASE_MODEL
    base_clf.fit(X_train_s, y_train)
    y_pred_none = base_clf.predict(X_test_s)
    cm_none = confusion_matrix(y_test, y_pred_none)
else:
    cm_none = None

cm_smote = None
if 'smote' in results:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_sm_tr, y_sm_tr = sm.fit_resample(X_train_s, y_train)
    sm_clf = BASE_MODEL
    sm_clf.fit(X_sm_tr, y_sm_tr)
    y_pred_sm = sm_clf.predict(X_test_s)
    cm_smote = confusion_matrix(y_test, y_pred_sm)

if cm_none is not None or cm_smote is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if cm_none is not None:
        sns.heatmap(cm_none, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix: Baseline (None)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
    else:
        axes[0].axis('off')
    if cm_smote is not None:
        sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
        axes[1].set_title('Confusion Matrix: SMOTE')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
    else:
        axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('visuals/augmentation/confusion_matrices_none_vs_smote.png', dpi=300, bbox_inches='tight')
    plt.close()

# Save metrics
import json
with open('reports/augmentation/metrics_by_technique.json', 'w') as f:
    json.dump(results, f, indent=2)

# Clinical plausibility checks & summary
summary_lines = []
summary_lines.append('# Phase 2 Augmentation Summary')
summary_lines.append('')
summary_lines.append('Techniques evaluated: none, random_over, SMOTE, BorderlineSMOTE, ADASYN, HbA1c-missing augmentation (train-only), Multi-Imputation augmentation (train-only)')
summary_lines.append('')
# Death rate baseline
death_rate = (y_raw.sum() / len(y_raw)) * 100
summary_lines.append(f'- Original death rate: {death_rate:.1f}% (class imbalance)')
# Report HbA1c missingness and augmentation details
if HBA1C_COL in df.columns:
    summary_lines.append(f'- HbA1c missing rate: {df[HBA1C_COL].isna().mean()*100:.1f}% in original dataset')
    if 'hba1c_aug' in results:
        try:
            miss_train_cnt = int(X_train_raw[HBA1C_COL].isna().sum())
            summary_lines.append(f'- HbA1c augmentation: created {miss_train_cnt * HBA1C_AUG_K} synthetic train rows ({HBA1C_AUG_K} draws per missing row) with clinically plausible values conditioned on diabetes/glucose status.')
        except Exception:
            pass
# Check if any augmentation overly distorts AUC or metrics
best_auc_name = max(results, key=lambda k: results[k]['auc'])
summary_lines.append(f'- Best AUC: {results[best_auc_name]["auc"]:.3f} with {best_auc_name}')
# Clinical sanity: compare medians for key features before vs after SMOTE
if 'smote_train' in class_dist:
   smote = SMOTE(random_state=RANDOM_STATE)
   X_sm, y_sm = smote.fit_resample(X_train, y_train)
   shifts = []
   for col in available_keys:
       med_orig = float(np.nanmedian(X_train[col]))
       med_sm = float(np.nanmedian(X_sm[col]))
       shift = med_sm - med_orig
       shifts.append((col, med_orig, med_sm, shift))
   summary_lines.append('\n## Feature Median Shifts (SMOTE)')
   for col, m0, m1, d in shifts:
       summary_lines.append(f'- {col}: {m0:.2f} -> {m1:.2f} (Δ {d:+.2f})')
   large_shifts = [s for s in shifts if abs(s[3]) > (0.25 * (np.nanstd(X_train[s[0]]) + 1e-6))]
   if large_shifts:
       summary_lines.append('\nWarning: Some features show notable median shifts after SMOTE. Verify clinical sense.')
else:
   summary_lines.append('\nSMOTE results unavailable for feature shift checks.')

summary_lines.append('\n## Interpretation Guidance')
summary_lines.append('- Improvements in recall may come at the cost of precision. Focus on F1 and AUC, not accuracy alone.')
summary_lines.append('- If augmentation substantially distorts key clinical feature distributions, treat results with caution.')

with open('reports/augmentation/augmentation_summary.md', 'w') as f:
   f.write('\n'.join(summary_lines))

print('Saved visuals to visuals/augmentation and report to reports/augmentation')
print('=== Phase 2 Augmentation Analysis Complete ===')