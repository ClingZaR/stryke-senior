#!/usr/bin/env python3
"""
Phase 2: Baseline Models & Preprocessing Pipeline
Mortality Risk Prediction - Stroke Dataset

This script implements:
- Data preprocessing and feature engineering
- Missing value imputation strategies
- CatBoost baseline model
- Comprehensive evaluation and visualization
- Model comparison and selection
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from catboost import CatBoostClassifier
import warnings
import os
from datetime import datetime
import json
import pickle

warnings.filterwarnings('ignore')
plt.style.use('default')

class Phase2BaselinePipeline:
    def __init__(self, data_path='../clean_data.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.label_encoders = {}
        self.scaler = None
        self.imputer = None
        
        # Create output directories
        os.makedirs('visuals', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
    def load_and_explore_data(self):
        """Load data and perform initial exploration"""
        print("=== Phase 2: Loading and Exploring Data ===")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully: {self.df.shape}")
            print(f"Features: {list(self.df.columns)}")
            
            # Check target distribution
            if 'stroke' in self.df.columns:
                target_dist = self.df['stroke'].value_counts()
                print(f"\nTarget Distribution:")
                print(target_dist)
                print(f"Class Imbalance Ratio: {target_dist[0]/target_dist[1]:.2f}:1")
                
                # Visualize target distribution
                plt.figure(figsize=(8, 6))
                target_dist.plot(kind='bar', color=['skyblue', 'salmon'])
                plt.title('Target Distribution (Stroke vs No Stroke)', fontsize=14, fontweight='bold')
                plt.xlabel('Stroke Status')
                plt.ylabel('Count')
                plt.xticks([0, 1], ['No Stroke', 'Stroke'], rotation=0)
                plt.tight_layout()
                plt.savefig('visuals/target_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Target distribution visualization saved")
                
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Comprehensive data preprocessing"""
        print("\n=== Data Preprocessing ===")
        
        # Derive/normalize target column
        if 'stroke' not in self.df.columns:
            # Common alternative naming found in dataset
            alt_targets = [
                'Death outcome (YES/NO)',
                'death outcome (YES/NO)',
                'Death outcome',
                'death_outcome',
                'outcome'
            ]
            for col in alt_targets:
                if col in self.df.columns:
                    # Map YES/NO or similar to 1/0
                    mapped = self.df[col].astype(str).str.strip().str.lower().map({
                        'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0
                    })
                    if mapped.notna().any():
                        self.df['stroke'] = mapped.fillna(0).astype(int)
                        print(f"Target column '{col}' mapped to binary 'stroke'.")
                        break
        
        # Separate features and target
        if 'stroke' not in self.df.columns:
            print("Error: 'stroke' column not found")
            return False
            
        # Drop obvious identifiers if present to reduce leakage risk
        drop_leak = [c for c in ['CMRN', 'Unnamed: 0'] if c in self.df.columns]
        if drop_leak:
            self.df = self.df.drop(columns=drop_leak)
            print(f"Dropped potential identifier columns: {drop_leak}")
            
        X = self.df.drop('stroke', axis=1)
        y = self.df['stroke']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"Categorical columns: {list(categorical_cols)}")
        
        # Label encode categorical variables
        X_encoded = X.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle missing values in categorical columns
            X_encoded[col] = X_encoded[col].fillna('Unknown')
            X_encoded[col] = le.fit_transform(X_encoded[col])
            self.label_encoders[col] = le
        
        # Handle missing values in numerical columns
        numerical_cols = X_encoded.select_dtypes(include=[np.number]).columns
        print(f"Numerical columns: {list(numerical_cols)}")
        
        # Check missing values
        missing_info = X_encoded.isnull().sum()
        if missing_info.sum() > 0:
            print(f"\nMissing values found:")
            print(missing_info[missing_info > 0])
            
            # Use Simple imputation for compatibility
            self.imputer = SimpleImputer(strategy='median')
            X_encoded = pd.DataFrame(
                self.imputer.fit_transform(X_encoded),
                columns=X_encoded.columns,
                index=X_encoded.index
            )
            print("Missing values imputed using median imputation")
        
        # Store feature names
        self.feature_names = list(X_encoded.columns)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        return True
    
    def train_baseline_models(self):
        """Train CatBoost baseline model"""
        print("\n=== Training Baseline Models ===")
        
        # Model configurations
        catboost_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'eval_metric': 'AUC',
            'early_stopping_rounds': 100,
            'class_weights': [1, 10]  # Handle class imbalance with weights
        }
        
        # Train CatBoost with class weights
        print("Training CatBoost (Class Weighted)...")
        self.models['catboost_weighted'] = CatBoostClassifier(**catboost_params)
        self.models['catboost_weighted'].fit(
            self.X_train_scaled, self.y_train,
            eval_set=(self.X_test_scaled, self.y_test),
            plot=False
        )
        
        # Train CatBoost without class weights for comparison
        catboost_params_normal = catboost_params.copy()
        del catboost_params_normal['class_weights']
        
        print("Training CatBoost (Standard)...")
        self.models['catboost_standard'] = CatBoostClassifier(**catboost_params_normal)
        self.models['catboost_standard'].fit(
            self.X_train_scaled, self.y_train,
            eval_set=(self.X_test_scaled, self.y_test),
            plot=False
        )
        
        print("Models trained successfully!")
        return True
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n=== Model Evaluation ===")
        
        for model_name, model in self.models.items():
            print(f"\n--- {model_name.upper()} ---")
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Metrics (test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)

            # Train-side metrics
            y_train_pred = model.predict(self.X_train_scaled)
            y_train_proba = model.predict_proba(self.X_train_scaled)[:, 1]
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            train_f1 = f1_score(self.y_train, y_train_pred)
            train_auc = roc_auc_score(self.y_train, y_train_proba)

            # 5-fold CV on training set (F1)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_f1_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='f1')
            cv_f1_mean = float(cv_f1_scores.mean())
            cv_f1_std = float(cv_f1_scores.std())
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC-ROC: {auc:.4f}")
            print(f"Train F1: {train_f1:.4f} | CV F1 (5-fold): {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
            
            # Store results
            self.results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_roc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'train_accuracy': train_accuracy,
                'train_f1': train_f1,
                'train_auc': train_auc,
                'cv_f1_mean': cv_f1_mean,
                'cv_f1_std': cv_f1_std
            }
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n=== Creating Visualizations ===")
        
        # 1. Model Performance Comparison
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results.keys()],
            'F1-Score': [self.results[m]['f1_score'] for m in self.results.keys()],
            'AUC-ROC': [self.results[m]['auc_roc'] for m in self.results.keys()]
        })
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(['Accuracy', 'F1-Score', 'AUC-ROC']):
            axes[i].bar(metrics_df['Model'], metrics_df[metric], 
                       color=['lightblue', 'lightcoral'])
            axes[i].set_title(f'{metric} Comparison', fontweight='bold')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(metrics_df[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('visuals/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Model performance comparison saved")
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        
        for model_name in self.results.keys():
            fpr, tpr, _ = roc_curve(self.y_test, self.results[model_name]['probabilities'])
            auc = self.results[model_name]['auc_roc']
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visuals/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ ROC curves comparison saved")
        
        # 3. Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        
        for model_name in self.results.keys():
            precision, recall, _ = precision_recall_curve(
                self.y_test, self.results[model_name]['probabilities']
            )
            plt.plot(recall, precision, linewidth=2, label=f'{model_name}')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visuals/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Precision-recall curves saved")
        
        # 4. Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, model_name in enumerate(self.results.keys()):
            cm = confusion_matrix(self.y_test, self.results[model_name]['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['No Stroke', 'Stroke'],
                       yticklabels=['No Stroke', 'Stroke'])
            axes[i].set_title(f'{model_name} Confusion Matrix', fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('visuals/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Confusion matrices saved")
        
        # 5. Feature Importance (for best model)
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['auc_roc'])
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'], 
                    color='skyblue')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importances - {best_model_name}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('visuals/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Feature importance plot saved")
        
        print("All visualizations created successfully!")
    
    def save_models_and_results(self):
        """Save models and results"""
        print("\n=== Saving Models and Results ===")
        
        # Save best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['auc_roc'])
        best_model = self.models[best_model_name]
        
        # Save model using CatBoost's native save method
        best_model.save_model('models/phase2_best_model.cbm')
        
        # Save preprocessing objects
        with open('models/phase2_preprocessors.pkl', 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'imputer': self.imputer,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names
            }, f)
        
        # Save results summary
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'model_performance': {
                model_name: {
                    'accuracy': float(results['accuracy']),
                    'f1_score': float(results['f1_score']),
                    'auc_roc': float(results['auc_roc'])
                }
                for model_name, results in self.results.items()
            },
            'data_info': {
                'total_samples': len(self.df),
                'features': len(self.feature_names),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test)
            }
        }
        
        with open('reports/phase2_results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"✓ Best model ({best_model_name}) saved successfully!")
        print(f"✓ Results summary saved to reports/phase2_results_summary.json")
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n=== Generating Phase 2 Report ===")
        
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['auc_roc'])
        
        report = f"""
# Phase 2: Baseline Models & Preprocessing Report

## Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset**: {self.data_path}
- **Total Samples**: {len(self.df)}
- **Features**: {len(self.feature_names)}
- **Train/Test Split**: {len(self.X_train)}/{len(self.X_test)}

## Data Preprocessing
- **Missing Value Handling**: Median Imputation
- **Categorical Encoding**: Label Encoding
- **Feature Scaling**: Standard Scaler
- **Class Imbalance**: Class weights (1:10 ratio)

## Model Performance

### CatBoost (Standard)
- **Accuracy**: {self.results['catboost_standard']['accuracy']:.4f}
- **F1-Score**: {self.results['catboost_standard']['f1_score']:.4f}
- **AUC-ROC**: {self.results['catboost_standard']['auc_roc']:.4f}

### CatBoost (Class Weighted)
- **Accuracy**: {self.results['catboost_weighted']['accuracy']:.4f}
- **F1-Score**: {self.results['catboost_weighted']['f1_score']:.4f}
- **AUC-ROC**: {self.results['catboost_weighted']['auc_roc']:.4f}

## Best Model
- **Selected Model**: {best_model_name}
- **Best AUC-ROC**: {self.results[best_model_name]['auc_roc']:.4f}

## Key Findings
1. {'Class weighting improved model performance' if self.results['catboost_weighted']['auc_roc'] > self.results['catboost_standard']['auc_roc'] else 'Standard model performed better than class weighting'}
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
"""
        
        with open('reports/phase2_report.md', 'w') as f:
            f.write(report)
        
        print("✓ Phase 2 report generated successfully!")
    
    def generate_overfitting_report(self):
        """Generate overfitting diagnostics and save to JSON"""
        print("\n=== Overfitting Analysis ===")

        report = {
            'timestamp': datetime.now().isoformat(),
            'criteria': {
                'f1_train_minus_test_threshold': 0.05,
                'auc_train_minus_test_threshold': 0.05,
                'test_vs_cv_z_threshold': 2.0
            },
            'models': {}
        }

        for name, res in self.results.items():
            f1_gap = float(res.get('train_f1', 0) - res.get('f1_score', 0))
            auc_gap = float(res.get('train_auc', 0) - res.get('auc_roc', 0))
            cv_mean = float(res.get('cv_f1_mean', 0))
            cv_std = float(res.get('cv_f1_std', 1e-9))
            test_f1 = float(res.get('f1_score', 0))
            z = (test_f1 - cv_mean) / (cv_std if cv_std > 0 else 1e-9)

            potential_overfit = (f1_gap > report['criteria']['f1_train_minus_test_threshold']) or \
                                 (auc_gap > report['criteria']['auc_train_minus_test_threshold'])
            suspicious_split = abs(z) > report['criteria']['test_vs_cv_z_threshold']

            verdict = 'pass'
            if potential_overfit:
                verdict = 'potential_overfit'
            if suspicious_split and not potential_overfit:
                verdict = 'suspicious_split_or_leakage'
            if suspicious_split and potential_overfit:
                verdict = 'overfit_and_suspicious_split'

            report['models'][name] = {
                'train_f1': res.get('train_f1'),
                'test_f1': res.get('f1_score'),
                'f1_gap': f1_gap,
                'train_auc': res.get('train_auc'),
                'test_auc': res.get('auc_roc'),
                'auc_gap': auc_gap,
                'cv_f1_mean': cv_mean,
                'cv_f1_std': cv_std,
                'test_vs_cv_z': z,
                'verdict': verdict
            }

            print(f"{name}: verdict={verdict} | train_test_f1_gap={f1_gap:.3f} | z(Test vs CV)={z:.2f}")

        out_path = 'reports/phase2_overfitting.json'
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Overfitting report saved to {out_path}")
    
    def run_complete_pipeline(self):
        """Run the complete Phase 2 pipeline"""
        print("\n" + "="*60)
        print("PHASE 2: BASELINE MODELS & PREPROCESSING PIPELINE")
        print("="*60)
        
        steps = [
            ("Loading Data", self.load_and_explore_data),
            ("Preprocessing", self.preprocess_data),
            ("Training Models", self.train_baseline_models),
            ("Evaluating Models", self.evaluate_models),
            ("Overfitting Analysis", self.generate_overfitting_report),
            ("Creating Visualizations", self.create_visualizations),
            ("Saving Results", self.save_models_and_results),
            ("Generating Report", self.generate_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            try:
                success = step_func()
                if success is False:
                    print(f"Error in {step_name}. Pipeline stopped.")
                    return False
            except Exception as e:
                print(f"Error in {step_name}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n" + "="*60)
        print("PHASE 2 COMPLETED SUCCESSFULLY!")
        print("="*60)
        return True

if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = Phase2BaselinePipeline()
    pipeline.run_complete_pipeline()