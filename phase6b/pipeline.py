"""
Phase 6b Pipeline: IST Database Integration with Enhanced Treatment Modeling

This pipeline integrates the International Stroke Trial (IST) database with the existing
stroke prediction system, adding treatment-specific features and modeling capabilities.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Import our custom modules
from .ist_integration import ISTDataHarmonizer, validate_integration
from .treatment_features import TreatmentFeatureEngineer, TreatmentOutcomeAnalyzer


class Phase6bPipeline:
    """
    Main pipeline for Phase 6b: IST integration and treatment modeling.
    """
    
    def __init__(self, data_path: str = None, ist_path: str = None):
        self.data_path = data_path or "../clean_data.csv"
        self.ist_path = ist_path or "ist_data.csv"  # Will be simulated if not found
        
        self.harmonizer = ISTDataHarmonizer()
        self.feature_engineer = TreatmentFeatureEngineer()
        self.outcome_analyzer = TreatmentOutcomeAnalyzer()
        
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
        
        # Enhanced feature set including IST variables
        self.enhanced_features = [
            'Age', 'Gender', 'BP_sys', 'BP_dia', 'BMI',
            'known case of atrial fibrillation (YES/NO)',
            'Known case of Hypertension', 'known case of diabetes',
            'Aspirin administered (YES/NO)', 'aspirin_administered',
            'heparin_administered', 'treatment_combination',
            'treatment_timing', 'early_treatment', 'delayed_treatment',
            'age_aspirin_interaction', 'stroke_risk_score'
        ]
        
    def load_and_harmonize_data(self) -> pd.DataFrame:
        """Load and harmonize clinical and IST datasets."""
        print("Loading and harmonizing datasets...")
        
        # Load clinical data
        try:
            clinical_data = pd.read_csv(self.data_path)
            print(f"Loaded clinical data: {clinical_data.shape}")
        except FileNotFoundError:
            print(f"Clinical data file not found: {self.data_path}")
            return None
        
        # Load and harmonize with IST data
        unified_data = self.harmonizer.load_and_harmonize(
            clinical_path=self.data_path,
            ist_path=self.ist_path
        )
        
        print(f"Unified dataset shape: {unified_data.shape}")
        print(f"Dataset sources: {unified_data['dataset_source'].value_counts()}")
        
        return unified_data
    
    def engineer_treatment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply treatment-specific feature engineering."""
        print("Engineering treatment features...")
        
        # Fit and transform treatment features
        self.feature_engineer.fit(df)
        df_enhanced = self.feature_engineer.transform(df)
        
        print(f"Features after engineering: {df_enhanced.shape[1]}")
        print("New treatment features created:")
        
        new_cols = set(df_enhanced.columns) - set(df.columns)
        for col in sorted(new_cols):
            print(f"  - {col}")
        
        return df_enhanced
    
    def prepare_modeling_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling with enhanced features."""
        print("Preparing modeling data...")
        
        # Define target
        target_col = 'Death outcome (YES/NO)'
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found. Available columns:")
            print(df.columns.tolist())
            return None, None
        
        # Prepare target variable
        y = df[target_col].copy()
        if y.dtype == 'object':
            y = (y.str.upper() == 'YES').astype(int)
        
        # Select features (use available features from enhanced set)
        available_features = [col for col in self.enhanced_features if col in df.columns]
        
        # Add any treatment-related columns that were created
        treatment_cols = [col for col in df.columns if any(
            term in col.lower() for term in [
                'treatment', 'aspirin', 'heparin', 'combo_', 'timing_',
                'interaction', 'risk_score', 'early_', 'delayed_'
            ]
        )]
        
        all_features = list(set(available_features + treatment_cols))
        X = df[all_features].copy()
        
        print(f"Selected {len(all_features)} features for modeling")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Handle missing values
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Impute missing values
        for col in numeric_features:
            X[col] = X[col].fillna(X[col].median())
        
        for col in categorical_features:
            X[col] = X[col].fillna('unknown')
            # Encode categorical variables
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_enhanced_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train models with enhanced treatment features."""
        print("Training enhanced models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42, learning_rate=0.1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for logistic regression, original for tree-based
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled if name == 'logistic_regression' else X_train,
                y_train, cv=5, scoring='roc_auc'
            )
            
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred_proba
            }
            
            print(f"AUC Score: {auc_score:.4f}")
            print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store model
            self.models[name] = model
        
        # Store test data for analysis
        self.test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': self.feature_names
        }
        
        return results
    
    def analyze_treatment_effectiveness(self, df: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze treatment effectiveness patterns."""
        print("Analyzing treatment effectiveness...")
        
        effectiveness_results = self.outcome_analyzer.analyze_treatment_effectiveness(df, y)
        
        print("\nTreatment Effectiveness Summary:")
        for treatment, results in effectiveness_results.items():
            if isinstance(results, dict) and 'treated_mortality' in results:
                print(f"\n{treatment}:")
                print(f"  Treated mortality: {results['treated_mortality']:.3f}")
                print(f"  Untreated mortality: {results['untreated_mortality']:.3f}")
                print(f"  Relative risk reduction: {results['relative_risk_reduction']:.3f}")
        
        return effectiveness_results
    
    def generate_feature_importance_report(self) -> Dict:
        """Generate feature importance analysis."""
        print("Generating feature importance report...")
        
        importance_report = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_[0])
            else:
                continue
            
            # Create importance dataframe
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importance_report[name] = feature_importance
            
            print(f"\nTop 10 features for {name}:")
            print(feature_importance.head(10).to_string(index=False))
        
        return importance_report
    
    def save_results(self, output_dir: str = "results") -> None:
        """Save all results and models."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving results to {output_dir}...")
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f"{name}_model.pkl"))
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(output_dir, f"{name}_scaler.pkl"))
        
        # Save feature names
        with open(os.path.join(output_dir, "feature_names.txt"), 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        # Save results summary
        if hasattr(self, 'results'):
            pd.DataFrame(self.results).to_csv(
                os.path.join(output_dir, "model_results.csv")
            )
        
        print("Results saved successfully!")
    
    def run_full_pipeline(self) -> Dict:
        """Run the complete Phase 6b pipeline."""
        print("=" * 60)
        print("PHASE 6B: IST INTEGRATION AND TREATMENT MODELING")
        print("=" * 60)
        
        # Step 1: Load and harmonize data
        unified_data = self.load_and_harmonize_data()
        if unified_data is None:
            return {"error": "Failed to load data"}
        
        # Step 2: Validate integration
        validation_results = validate_integration(unified_data)
        print(f"\nData validation: {validation_results}")
        
        # Step 3: Engineer treatment features
        enhanced_data = self.engineer_treatment_features(unified_data)
        
        # Step 4: Prepare modeling data
        X, y = self.prepare_modeling_data(enhanced_data)
        if X is None:
            return {"error": "Failed to prepare modeling data"}
        
        # Step 5: Train enhanced models
        model_results = self.train_enhanced_models(X, y)
        
        # Step 6: Analyze treatment effectiveness
        treatment_analysis = self.analyze_treatment_effectiveness(enhanced_data, y)
        
        # Step 7: Generate feature importance report
        importance_report = self.generate_feature_importance_report()
        
        # Step 8: Save results
        self.save_results()
        
        # Compile final results
        final_results = {
            'data_shape': enhanced_data.shape,
            'model_performance': {
                name: {
                    'auc_score': results['auc_score'],
                    'cv_mean': results['cv_mean'],
                    'cv_std': results['cv_std']
                }
                for name, results in model_results.items()
            },
            'treatment_effectiveness': treatment_analysis,
            'feature_importance': importance_report,
            'validation': validation_results
        }
        
        self.results = final_results
        
        print("\n" + "=" * 60)
        print("PHASE 6B PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return final_results


def main():
    """Main execution function."""
    # Initialize pipeline
    pipeline = Phase6bPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline()
    
    # Print summary
    if 'error' not in results:
        print("\nPIPELINE SUMMARY:")
        print(f"Dataset shape: {results['data_shape']}")
        print("\nModel Performance:")
        for model, metrics in results['model_performance'].items():
            print(f"  {model}: AUC = {metrics['auc_score']:.4f}")
        
        print(f"\nResults saved to: results/")
        print("Phase 6b integration completed successfully!")
    else:
        print(f"Pipeline failed: {results['error']}")


if __name__ == "__main__":
    main()