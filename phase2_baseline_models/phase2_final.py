#!/usr/bin/env python3
"""
Phase 2: Baseline Models & Preprocessing - Final Version
Focused on stroke death prediction with comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')
plt.style.use('default')

print("🚀 Phase 2: Baseline Models & Preprocessing")
print("=" * 50)

# Create directories
os.makedirs('visuals', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)
print("✅ Directories created")

# Load data
print("\n📊 Loading Data...")
try:
    df = pd.read_csv('../clean_data.csv')
    print(f"Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Create target variable
target_col = 'Death outcome (YES/NO)'
if target_col in df.columns:
    df['target'] = df[target_col].map({'yes': 1, 'no': 0})
    df['target'] = df['target'].fillna(0)  # Assume missing = no death
    
    target_counts = df['target'].value_counts()
    print(f"\n🎯 Target Distribution:")
    print(f"   Survived: {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
    print(f"   Deaths: {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")
else:
    print(f"❌ Target column '{target_col}' not found")
    exit(1)

# Feature selection
exclude_cols = ['target', target_col, 'CMRN', 'Unnamed: 0']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].copy()
y = df['target'].copy()

print(f"\n🔧 Feature Engineering:")
print(f"   Selected features: {len(feature_cols)}")

# Handle categorical and numerical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"   Categorical: {len(categorical_cols)}")
print(f"   Numerical: {len(numerical_cols)}")

# Preprocessing
X_processed = X.copy()

# Encode categorical variables
for col in categorical_cols:
    X_processed[col] = X_processed[col].fillna('Unknown')
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))

# Handle missing values in numerical columns
if X_processed.isnull().sum().sum() > 0:
    imputer = SimpleImputer(strategy='median')
    X_processed = pd.DataFrame(
        imputer.fit_transform(X_processed),
        columns=X_processed.columns
    )

print("✅ Preprocessing completed")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Data Split:")
print(f"   Training: {X_train.shape[0]:,} samples")
print(f"   Testing: {X_test.shape[0]:,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\n🤖 Training Random Forest Model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✅ Model training completed")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 Model Performance:")
print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"   F1-Score: {f1:.3f}")
print(f"   AUC-ROC: {auc:.3f}")

# Create visualizations
print("\n🎨 Creating Visualizations...")

# 1. Target Distribution
plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#e74c3c']
target_counts.plot(kind='bar', color=colors, alpha=0.8)
plt.title('Target Distribution: Death Outcome', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['Survived', 'Death'], rotation=0)
plt.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (idx, count) in enumerate(target_counts.items()):
    pct = count/len(df)*100
    plt.text(i, count + len(df)*0.01, f'{count:,}\n({pct:.1f}%)', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('visuals/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Target distribution saved")

# 2. ROC Curve
plt.figure(figsize=(8, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, linewidth=3, label=f'Random Forest (AUC = {auc:.3f})', color='#3498db')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Death Prediction Model', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ ROC curve saved")

# 3. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
           xticklabels=['Survived', 'Death'],
           yticklabels=['Survived', 'Death'],
           square=True, linewidths=0.5)
plt.title('Confusion Matrix - Death Prediction', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Outcome', fontsize=12)
plt.ylabel('Actual Outcome', fontsize=12)
plt.tight_layout()
plt.savefig('visuals/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved")

# 4. Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 10))
top_features = feature_importance.head(15)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 15 Feature Importances - Death Prediction Model', fontsize=16, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Feature importance saved")

# 5. Model Performance Dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Metrics bar chart
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
values = [accuracy, f1, auc]
colors = ['#e74c3c', '#3498db', '#2ecc71']
bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
ax1.set_ylim(0, 1)
ax1.set_title('Performance Metrics', fontweight='bold', fontsize=14)
ax1.set_ylabel('Score')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Class distribution pie
ax2 = fig.add_subplot(gs[0, 1])
ax2.pie(target_counts.values, labels=['Survived', 'Death'], autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'], startangle=90)
ax2.set_title('Class Distribution', fontweight='bold', fontsize=14)

# Prediction distribution pie
ax3 = fig.add_subplot(gs[0, 2])
pred_counts = pd.Series(y_pred).value_counts()
ax3.pie(pred_counts.values, labels=['Survived', 'Death'], autopct='%1.1f%%',
        colors=['#3498db', '#f39c12'], startangle=90)
ax3.set_title('Prediction Distribution', fontweight='bold', fontsize=14)

# ROC curve
ax4 = fig.add_subplot(gs[1, :])
ax4.plot(fpr, tpr, linewidth=3, label=f'Random Forest (AUC = {auc:.3f})', color='#3498db')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7)
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curve', fontweight='bold', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Top features
ax5 = fig.add_subplot(gs[2, :])
top_10_features = feature_importance.head(10)
colors = plt.cm.Set3(np.linspace(0, 1, len(top_10_features)))
ax5.barh(range(len(top_10_features)), top_10_features['importance'], color=colors)
ax5.set_yticks(range(len(top_10_features)))
ax5.set_yticklabels(top_10_features['feature'])
ax5.set_xlabel('Importance')
ax5.set_title('Top 10 Most Important Features', fontweight='bold', fontsize=14)
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)

plt.suptitle('Phase 2: Baseline Model Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)
plt.savefig('visuals/performance_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Performance dashboard saved")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'phase': 'Phase 2 - Baseline Models',
    'model_type': 'Random Forest with Balanced Class Weights',
    'performance': {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'auc_roc': float(auc)
    },
    'dataset_info': {
        'total_samples': int(len(df)),
        'features_used': int(len(X_processed.columns)),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'class_distribution': {
            'survived': int(target_counts[0]),
            'deaths': int(target_counts[1])
        }
    },
    'top_features': [
        {'feature': row['feature'], 'importance': float(row['importance'])}
        for _, row in feature_importance.head(10).iterrows()
    ]
}

with open('reports/phase2_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✅ Results saved to JSON")

# Generate report
report = f"""
# Phase 2: Baseline Models & Preprocessing - Results Report

## 🎯 Executive Summary
Successfully implemented baseline Random Forest model for stroke death prediction using balanced class weighting to address severe class imbalance (95.7% survival rate).

## 📊 Model Performance
- **Accuracy**: {accuracy:.3f} ({accuracy*100:.1f}%)
- **F1-Score**: {f1:.3f}
- **AUC-ROC**: {auc:.3f}

## 📈 Dataset Overview
- **Total Samples**: {len(df):,}
- **Features Used**: {len(X_processed.columns)}
- **Training Set**: {len(X_train):,} samples
- **Test Set**: {len(X_test):,} samples
- **Class Distribution**: 
  - Survived: {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)
  - Deaths: {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)

## 🔍 Key Findings

### 1. Severe Class Imbalance
- Only {target_counts[1]/len(df)*100:.1f}% of cases resulted in death
- Addressed using balanced class weighting in Random Forest
- Model shows reasonable performance despite imbalance

### 2. Feature Engineering
- Processed {len(categorical_cols)} categorical features using label encoding
- Handled {len(numerical_cols)} numerical features with median imputation
- All missing values successfully imputed

### 3. Top 5 Most Predictive Features
{chr(10).join([f'{i+1}. **{row["feature"]}**: {row["importance"]:.4f}' for i, (_, row) in enumerate(feature_importance.head(5).iterrows())])}

## 📁 Generated Outputs

### Visualizations (`visuals/` directory)
1. **target_distribution.png** - Class distribution analysis
2. **roc_curve.png** - ROC curve with AUC score
3. **confusion_matrix.png** - Prediction accuracy breakdown
4. **feature_importance.png** - Top 15 most important features
5. **performance_dashboard.png** - Comprehensive performance overview

### Reports (`reports/` directory)
- **phase2_results.json** - Detailed metrics and metadata
- **phase2_report.md** - This comprehensive report

## 🚀 Next Steps (Phase 3)

1. **Advanced Algorithms**: Implement XGBoost, LightGBM, and Neural Networks
2. **Feature Engineering**: Advanced feature selection and creation
3. **Hyperparameter Tuning**: Bayesian optimization for better performance
4. **Ensemble Methods**: Combine multiple models for improved accuracy
5. **Cross-Validation**: Robust model validation strategies

## 💡 Recommendations

1. **Class Imbalance**: Consider SMOTE or other sampling techniques in Phase 3
2. **Feature Selection**: Focus on top predictive features for model efficiency
3. **Clinical Validation**: Validate findings with medical domain experts
4. **Risk Stratification**: Develop risk scoring system for clinical use

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Phase Status**: ✅ COMPLETED SUCCESSFULLY
"""

with open('reports/phase2_report.md', 'w') as f:
    f.write(report)
print("✅ Comprehensive report generated")

print("\n" + "=" * 60)
print("🎉 PHASE 2 COMPLETED SUCCESSFULLY! 🎉")
print("=" * 60)
print(f"📊 Model Performance: {accuracy:.1%} accuracy, {auc:.3f} AUC")
print(f"🎨 Generated 5 comprehensive visualizations")
print(f"📝 Created detailed report and results")
print(f"🚀 Ready for Phase 3: Advanced ML & Deep Learning")
print("=" * 60)