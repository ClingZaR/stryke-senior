#!/usr/bin/env python3
"""
Generate SMOTE Comparison Visualizations for Phase 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("Starting SMOTE Comparison Analysis...")

# Load and preprocess data
print("Loading data...")
data = pd.read_csv('../clean_data.csv')
print(f"Dataset shape: {data.shape}")

# Check death outcome distribution
death_col = 'Death outcome (YES/NO)'
if death_col in data.columns:
    death_counts = data[death_col].value_counts()
    print(f"Death outcome distribution: {death_counts.to_dict()}")
    death_rate = death_counts.get('yes', 0) / len(data) * 100
    print(f"Death rate: {death_rate:.1f}%")
else:
    print("Death outcome column not found!")
    exit(1)

# Prepare features
feature_cols = [col for col in data.columns if col not in [
    'Death outcome (YES/NO)', 'CMRN', 'Unnamed: 0'
]]

X = data[feature_cols].copy()
y = data[death_col].copy()

print(f"Features: {len(feature_cols)}")
print(f"Missing values: {X.isnull().sum().sum()}")

# Encode categorical variables
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encode target
y_encoded = le.fit_transform(y.astype(str))

print(f"Preprocessed data shape: {X.shape}")
print(f"Target distribution: {pd.Series(y_encoded).value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {len(y_train)} samples")
print(f"Test set: {len(y_test)} samples")

# Train Random Forest without SMOTE
print("\nTraining Random Forest without SMOTE...")
rf_no_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_no_smote.fit(X_train_scaled, y_train)
y_pred_no_smote = rf_no_smote.predict(X_test_scaled)
y_prob_no_smote = rf_no_smote.predict_proba(X_test_scaled)[:, 1]
auc_no_smote = roc_auc_score(y_test, y_prob_no_smote)
report_no_smote = classification_report(y_test, y_pred_no_smote, output_dict=True)

print(f"AUC without SMOTE: {auc_no_smote:.3f}")

# Apply SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE: {len(y_train_smote)} samples")
print(f"SMOTE class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

# Train Random Forest with SMOTE
print("\nTraining Random Forest with SMOTE...")
rf_with_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_with_smote.fit(X_train_smote, y_train_smote)
y_pred_with_smote = rf_with_smote.predict(X_test_scaled)
y_prob_with_smote = rf_with_smote.predict_proba(X_test_scaled)[:, 1]
auc_with_smote = roc_auc_score(y_test, y_prob_with_smote)
report_with_smote = classification_report(y_test, y_pred_with_smote, output_dict=True)

print(f"AUC with SMOTE: {auc_with_smote:.3f}")

# Create comprehensive visualizations
print("\nGenerating visualizations...")

# Set up the plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Phase 2: Comprehensive SMOTE vs No-SMOTE Analysis', fontsize=16, fontweight='bold')

# 1. Class Distribution Comparison
ax1 = axes[0, 0]
original_dist = pd.Series(y_encoded).value_counts()
smote_dist = pd.Series(y_train_smote).value_counts()

x = np.arange(2)
width = 0.35
ax1.bar(x - width/2, [original_dist[0], original_dist[1]], width, label='Original', alpha=0.8, color='lightblue')
ax1.bar(x + width/2, [smote_dist[0], smote_dist[1]], width, label='After SMOTE', alpha=0.8, color='lightcoral')
ax1.set_xlabel('Class')
ax1.set_ylabel('Count')
ax1.set_title('Class Distribution: Original vs SMOTE')
ax1.legend()
ax1.set_xticks(x)
ax1.set_xticklabels(['Alive (0)', 'Death (1)'])

# 2. AUC Comparison
ax2 = axes[0, 1]
aucs = [auc_no_smote, auc_with_smote]
labels = ['No SMOTE', 'With SMOTE']
colors = ['lightblue', 'lightcoral']
bars = ax2.bar(labels, aucs, color=colors, alpha=0.8)
ax2.set_ylabel('AUC Score')
ax2.set_title('AUC Comparison')
ax2.set_ylim(0, 1)
for i, v in enumerate(aucs):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# 3. ROC Curves
ax3 = axes[0, 2]
fpr_no_smote, tpr_no_smote, _ = roc_curve(y_test, y_prob_no_smote)
fpr_with_smote, tpr_with_smote, _ = roc_curve(y_test, y_prob_with_smote)

ax3.plot(fpr_no_smote, tpr_no_smote, color='blue', label=f'No SMOTE (AUC={auc_no_smote:.3f})')
ax3.plot(fpr_with_smote, tpr_with_smote, color='red', label=f'With SMOTE (AUC={auc_with_smote:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Metrics Comparison
ax4 = axes[1, 0]
metrics = ['Precision\n(Death)', 'Recall\n(Death)', 'F1-Score\n(Death)']
no_smote_values = [
    report_no_smote['1']['precision'],
    report_no_smote['1']['recall'],
    report_no_smote['1']['f1-score']
]
with_smote_values = [
    report_with_smote['1']['precision'],
    report_with_smote['1']['recall'],
    report_with_smote['1']['f1-score']
]

x = np.arange(len(metrics))
ax4.bar(x - width/2, no_smote_values, width, label='No SMOTE', alpha=0.8, color='lightblue')
ax4.bar(x + width/2, with_smote_values, width, label='With SMOTE', alpha=0.8, color='lightcoral')
ax4.set_xlabel('Metrics')
ax4.set_ylabel('Score')
ax4.set_title('Detailed Metrics Comparison (Death Class)')
ax4.legend()
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.set_ylim(0, 1)

# 5. Confusion Matrix - No SMOTE
ax5 = axes[1, 1]
cm_no_smote = confusion_matrix(y_test, y_pred_no_smote)
sns.heatmap(cm_no_smote, annot=True, fmt='d', cmap='Blues', ax=ax5)
ax5.set_title('Confusion Matrix: No SMOTE')
ax5.set_xlabel('Predicted')
ax5.set_ylabel('Actual')

# 6. Confusion Matrix - With SMOTE
ax6 = axes[1, 2]
cm_with_smote = confusion_matrix(y_test, y_pred_with_smote)
sns.heatmap(cm_with_smote, annot=True, fmt='d', cmap='Oranges', ax=ax6)
ax6.set_title('Confusion Matrix: With SMOTE')
ax6.set_xlabel('Predicted')
ax6.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('visuals/comprehensive_smote_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visuals/comprehensive_smote_analysis.png")
plt.close()

# Create accuracy explanation plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Why High Accuracy is Misleading in Imbalanced Data', fontsize=14, fontweight='bold')

# Left plot: Class imbalance pie chart
ax1 = axes[0]
death_counts = pd.Series(y_encoded).value_counts()
colors = ['lightgreen', 'salmon']
wedges, texts, autotexts = ax1.pie(death_counts.values, labels=['Alive', 'Death'], 
                                  autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title(f'Severe Class Imbalance\n(Death Rate: {death_rate:.1f}%)')

# Right plot: Accuracy vs other metrics
ax2 = axes[1]
metrics = ['Accuracy', 'Precision\n(Death)', 'Recall\n(Death)', 'F1\n(Death)']
no_smote_all = [
    report_no_smote['accuracy'],
    report_no_smote['1']['precision'],
    report_no_smote['1']['recall'],
    report_no_smote['1']['f1-score']
]
with_smote_all = [
    report_with_smote['accuracy'],
    report_with_smote['1']['precision'],
    report_with_smote['1']['recall'],
    report_with_smote['1']['f1-score']
]

x = np.arange(len(metrics))
width = 0.35
ax2.bar(x - width/2, no_smote_all, width, label='No SMOTE', alpha=0.8, color='lightblue')
ax2.bar(x + width/2, with_smote_all, width, label='With SMOTE', alpha=0.8, color='lightcoral')
ax2.set_xlabel('Metrics')
ax2.set_ylabel('Score')
ax2.set_title('Metric Comparison: Focus Beyond Accuracy')
ax2.legend()
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visuals/accuracy_explanation.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visuals/accuracy_explanation.png")
plt.close()

# Generate summary report
report_content = f"""
# Phase 2: SMOTE Analysis Summary

## Dataset Overview
- Total samples: {len(y_encoded)}
- Death rate: {death_rate:.1f}%
- Features: {len(feature_cols)}

## Model Performance Comparison

### Random Forest Results:

**Without SMOTE:**
- AUC: {auc_no_smote:.3f}
- Accuracy: {report_no_smote['accuracy']:.3f}
- Precision (Death): {report_no_smote['1']['precision']:.3f}
- Recall (Death): {report_no_smote['1']['recall']:.3f}
- F1-Score (Death): {report_no_smote['1']['f1-score']:.3f}

**With SMOTE:**
- AUC: {auc_with_smote:.3f}
- Accuracy: {report_with_smote['accuracy']:.3f}
- Precision (Death): {report_with_smote['1']['precision']:.3f}
- Recall (Death): {report_with_smote['1']['recall']:.3f}
- F1-Score (Death): {report_with_smote['1']['f1-score']:.3f}

## Key Insights

1. **High Accuracy Explanation**: The {report_no_smote['accuracy']*100:.1f}% accuracy is misleading due to {100-death_rate:.1f}% of cases being survival (class imbalance).

2. **SMOTE Impact**: 
   - {'Improved' if auc_with_smote > auc_no_smote else 'Reduced'} AUC by {abs(auc_with_smote - auc_no_smote):.3f}
   - {'Improved' if report_with_smote['1']['recall'] > report_no_smote['1']['recall'] else 'Reduced'} recall for death detection
   - {'Improved' if report_with_smote['1']['f1-score'] > report_no_smote['1']['f1-score'] else 'Reduced'} F1-score

3. **Clinical Relevance**: Focus on recall and F1-score for mortality prediction rather than accuracy.

## Visualizations Generated

1. `comprehensive_smote_analysis.png` - Complete SMOTE comparison
2. `accuracy_explanation.png` - Why accuracy is misleading

---
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('reports/smote_analysis_summary.md', 'w') as f:
    f.write(report_content)

print("✅ Saved: reports/smote_analysis_summary.md")

print("\n" + "="*60)
print("SMOTE ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("📊 visuals/comprehensive_smote_analysis.png")
print("📊 visuals/accuracy_explanation.png")
print("📄 reports/smote_analysis_summary.md")
print(f"\n🔍 Key finding: SMOTE {'improved' if auc_with_smote > auc_no_smote else 'reduced'} AUC from {auc_no_smote:.3f} to {auc_with_smote:.3f}")
print(f"💡 High accuracy ({report_no_smote['accuracy']*100:.1f}%) is due to {100-death_rate:.1f}% survival rate (class imbalance)")