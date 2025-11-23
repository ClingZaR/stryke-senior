"""
Treatment Feature Engineering for Phase 6b

This module provides specialized feature engineering for treatment-related variables
from the IST database integration, including treatment combinations, timing effects,
and interaction features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


class TreatmentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Specialized feature engineering for treatment-related variables.
    """
    
    def __init__(self, create_interactions: bool = True, 
                 timing_thresholds: Dict[str, float] = None):
        self.create_interactions = create_interactions
        self.timing_thresholds = timing_thresholds or {'early': 1.0, 'delayed': 2.0}
        self.treatment_encoders = {}
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the treatment feature engineer."""
        self.feature_names_in_ = X.columns.tolist()
        
        # Fit encoders for categorical treatment variables
        treatment_cols = [col for col in X.columns if 'treatment' in col.lower()]
        for col in treatment_cols:
            if X[col].dtype == 'object':
                encoder = LabelEncoder()
                encoder.fit(X[col].fillna('unknown').astype(str))
                self.treatment_encoders[col] = encoder
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataset with treatment features."""
        if not self.fitted:
            raise ValueError("TreatmentFeatureEngineer must be fitted before transform")
            
        X_transformed = X.copy()
        
        # 1. Treatment combination features
        X_transformed = self._create_treatment_combinations(X_transformed)
        
        # 2. Treatment timing features
        X_transformed = self._create_timing_features(X_transformed)
        
        # 3. Treatment interaction features
        if self.create_interactions:
            X_transformed = self._create_interaction_features(X_transformed)
        
        # 4. Treatment response patterns
        X_transformed = self._create_response_patterns(X_transformed)
        
        return X_transformed
    
    def _create_treatment_combinations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create treatment combination features."""
        X_combo = X.copy()
        
        # Check for aspirin and heparin columns
        aspirin_cols = [col for col in X.columns if 'aspirin' in col.lower()]
        heparin_cols = [col for col in X.columns if 'heparin' in col.lower()]
        
        # Create binary indicators if we have the columns
        if aspirin_cols:
            aspirin_col = aspirin_cols[0]
            X_combo['aspirin_binary'] = (X_combo[aspirin_col] == 1).astype(int)
        
        if heparin_cols:
            heparin_col = heparin_cols[0]
            X_combo['heparin_binary'] = (X_combo[heparin_col] == 1).astype(int)
        
        # Create combination categories if we have both
        if aspirin_cols and heparin_cols:
            aspirin_col = aspirin_cols[0]
            heparin_col = heparin_cols[0]
            
            # Create combination categories
            def determine_combo(row):
                aspirin = row['aspirin_binary'] == 1
                heparin = row['heparin_binary'] == 1
                
                if aspirin and heparin:
                    return 'aspirin_heparin'
                elif aspirin:
                    return 'aspirin_only'
                elif heparin:
                    return 'heparin_only'
                else:
                    return 'neither'
            
            X_combo['treatment_combination_derived'] = X_combo.apply(determine_combo, axis=1)
            
            # One-hot encode combinations
            combo_dummies = pd.get_dummies(X_combo['treatment_combination_derived'], 
                                         prefix='combo', dummy_na=False)
            X_combo = pd.concat([X_combo, combo_dummies], axis=1)
        
        return X_combo
    
    def _create_timing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create treatment timing features."""
        X_timing = X.copy()
        
        timing_cols = [col for col in X.columns if 'timing' in col.lower()]
        
        for timing_col in timing_cols:
            if timing_col in X_timing.columns:
                # Create categorical timing features
                X_timing['early_treatment'] = (
                    X_timing[timing_col] <= self.timing_thresholds['early']
                ).astype(int)
                
                X_timing['delayed_treatment'] = (
                    X_timing[timing_col] > self.timing_thresholds['delayed']
                ).astype(int)
                
                X_timing['optimal_timing'] = (
                    (X_timing[timing_col] > 0) & 
                    (X_timing[timing_col] <= self.timing_thresholds['early'])
                ).astype(int)
                
                # Create timing bins
                X_timing['timing_category'] = pd.cut(
                    X_timing[timing_col], 
                    bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
                    labels=['immediate', 'early', 'standard', 'delayed']
                )
                
                # One-hot encode timing categories
                timing_dummies = pd.get_dummies(X_timing['timing_category'], 
                                              prefix='timing', dummy_na=True)
                X_timing = pd.concat([X_timing, timing_dummies], axis=1)
        
        return X_timing
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create treatment interaction features."""
        X_interact = X.copy()
        
        # Age interactions
        if 'Age' in X_interact.columns or 'age' in X_interact.columns:
            age_col = 'Age' if 'Age' in X_interact.columns else 'age'
            
            # Age × Aspirin - look for binary columns first, then original columns
            aspirin_cols = [col for col in X_interact.columns if 'aspirin' in col.lower() and 'binary' in col]
            if not aspirin_cols:
                aspirin_cols = [col for col in X_interact.columns if 'aspirin' in col.lower()]
            
            if aspirin_cols:
                X_interact['age_aspirin_interaction'] = (
                    X_interact[age_col] * X_interact[aspirin_cols[0]]
                )
            
            # Age × Heparin - look for binary columns first, then original columns
            heparin_cols = [col for col in X_interact.columns if 'heparin' in col.lower() and 'binary' in col]
            if not heparin_cols:
                heparin_cols = [col for col in X_interact.columns if 'heparin' in col.lower()]
            
            if heparin_cols:
                X_interact['age_heparin_interaction'] = (
                    X_interact[age_col] * X_interact[heparin_cols[0]]
                )
            
            # Age categories for treatment stratification
            X_interact['age_category'] = pd.cut(
                X_interact[age_col],
                bins=[0, 65, 75, 85, np.inf],
                labels=['young', 'middle', 'elderly', 'very_elderly']
            )
        
        # Gender interactions
        if 'Gender' in X_interact.columns or 'gender' in X_interact.columns:
            gender_col = 'Gender' if 'Gender' in X_interact.columns else 'gender'
            
            # Encode gender if it's categorical
            if X_interact[gender_col].dtype == 'object':
                X_interact['gender_binary'] = (X_interact[gender_col].str.lower() == 'male').astype(int)
            else:
                X_interact['gender_binary'] = X_interact[gender_col]
            
            # Gender × Treatment interactions
            aspirin_cols = [col for col in X_interact.columns if 'aspirin' in col.lower() and 'binary' in col]
            if not aspirin_cols:
                aspirin_cols = [col for col in X_interact.columns if 'aspirin' in col.lower()]
            
            if aspirin_cols:
                X_interact['gender_aspirin_interaction'] = (
                    X_interact['gender_binary'] * X_interact[aspirin_cols[0]]
                )
        
        # Blood pressure interactions
        bp_cols = ['BP_sys', 'BP_dia']
        for bp_col in bp_cols:
            if bp_col in X_interact.columns:
                aspirin_cols = [col for col in X_interact.columns if 'aspirin' in col.lower() and 'binary' in col]
                if aspirin_cols:
                    X_interact[f'{bp_col.lower()}_aspirin_interaction'] = (
                        X_interact[bp_col] * X_interact[aspirin_cols[0]]
                    )
        
        return X_interact
    
    def _create_response_patterns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create treatment response pattern features."""
        X_response = X.copy()
        
        # Risk stratification for treatment selection
        risk_factors = ['Age', 'BP_sys', 'known case of atrial fibrillation (YES/NO)']
        available_risk_factors = [col for col in risk_factors if col in X_response.columns]
        
        if available_risk_factors:
            # Create composite risk score
            risk_score = 0
            
            if 'Age' in X_response.columns:
                risk_score += (X_response['Age'] > 75).astype(int)
            
            if 'BP_sys' in X_response.columns:
                risk_score += (X_response['BP_sys'] > 160).astype(int)
            
            if 'known case of atrial fibrillation (YES/NO)' in X_response.columns:
                risk_score += X_response['known case of atrial fibrillation (YES/NO)'].fillna(0)
            
            X_response['stroke_risk_score'] = risk_score
            
            # Risk-based treatment recommendations
            X_response['high_risk_patient'] = (risk_score >= 2).astype(int)
            X_response['anticoagulation_candidate'] = (
                (risk_score >= 1) & 
                (X_response.get('BP_sys', 0) < 180)  # Not too high BP
            ).astype(int)
        
        # Treatment contraindication flags
        if 'BP_sys' in X_response.columns:
            X_response['hypertension_severe'] = (X_response['BP_sys'] > 180).astype(int)
            X_response['aspirin_contraindication'] = (X_response['BP_sys'] > 200).astype(int)
        
        return X_response

    def create_treatment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create treatment features from a dataframe.
        """
        self.fit(df)
        return self.transform(df)


class TreatmentOutcomeAnalyzer:
    """
    Analyze treatment outcomes and effectiveness patterns.
    """
    
    def __init__(self):
        self.outcome_patterns = {}
        self.effectiveness_scores = {}
    
    def analyze_treatment_effectiveness(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Analyze treatment effectiveness across different patient groups.
        """
        results = {}
        
        # Overall treatment effectiveness
        treatment_cols = [col for col in X.columns if any(
            term in col.lower() for term in ['aspirin', 'heparin', 'treatment']
        )]
        
        for col in treatment_cols:
            if col in X.columns and X[col].dtype in ['int64', 'float64']:
                treated = y[X[col] == 1]
                untreated = y[X[col] == 0]
                
                if len(treated) > 0 and len(untreated) > 0:
                    results[f'{col}_effectiveness'] = {
                        'treated_mortality': treated.mean(),
                        'untreated_mortality': untreated.mean(),
                        'relative_risk_reduction': 1 - (treated.mean() / untreated.mean()) if untreated.mean() > 0 else 0,
                        'sample_sizes': {'treated': len(treated), 'untreated': len(untreated)}
                    }
        
        # Age-stratified effectiveness
        if 'Age' in X.columns:
            age_groups = pd.cut(X['Age'], bins=[0, 65, 75, 85, np.inf], 
                              labels=['<65', '65-74', '75-84', '85+'])
            
            for treatment_col in treatment_cols:
                if treatment_col in X.columns and X[treatment_col].dtype in ['int64', 'float64']:
                    age_effectiveness = {}
                    for age_group in age_groups.cat.categories:
                        mask = age_groups == age_group
                        if mask.sum() > 10:  # Minimum sample size
                            treated_mask = mask & (X[treatment_col] == 1)
                            untreated_mask = mask & (X[treatment_col] == 0)
                            
                            if treated_mask.sum() > 0 and untreated_mask.sum() > 0:
                                treated_outcome = y[treated_mask].mean()
                                untreated_outcome = y[untreated_mask].mean()
                                
                                age_effectiveness[str(age_group)] = {
                                    'treated_mortality': treated_outcome,
                                    'untreated_mortality': untreated_outcome,
                                    'benefit': untreated_outcome - treated_outcome
                                }
                    
                    results[f'{treatment_col}_by_age'] = age_effectiveness
        
        return results
    
    def generate_treatment_recommendations(self, patient_features: Dict) -> Dict:
        """
        Generate personalized treatment recommendations based on patient features.
        """
        recommendations = {
            'aspirin': {'recommended': False, 'confidence': 0.0, 'reasoning': []},
            'heparin': {'recommended': False, 'confidence': 0.0, 'reasoning': []},
            'combination': {'recommended': False, 'confidence': 0.0, 'reasoning': []}
        }
        
        age = patient_features.get('age', 0)
        bp_sys = patient_features.get('bp_sys', 0)
        atrial_fib = patient_features.get('atrial_fibrillation', 0)
        
        # Aspirin recommendations
        aspirin_score = 0.5  # Base probability
        
        if age > 65:
            aspirin_score += 0.2
            recommendations['aspirin']['reasoning'].append('Age > 65 increases benefit')
        
        if atrial_fib:
            aspirin_score += 0.3
            recommendations['aspirin']['reasoning'].append('Atrial fibrillation increases stroke risk')
        
        if bp_sys > 180:
            aspirin_score -= 0.4
            recommendations['aspirin']['reasoning'].append('High BP increases bleeding risk')
        
        recommendations['aspirin']['recommended'] = aspirin_score > 0.6
        recommendations['aspirin']['confidence'] = min(aspirin_score, 1.0)
        
        # Heparin recommendations (more conservative)
        heparin_score = 0.3  # Lower base probability
        
        if atrial_fib and age > 70:
            heparin_score += 0.4
            recommendations['heparin']['reasoning'].append('High-risk profile for cardioembolic stroke')
        
        if bp_sys > 160:
            heparin_score -= 0.3
            recommendations['heparin']['reasoning'].append('Elevated BP increases bleeding risk')
        
        recommendations['heparin']['recommended'] = heparin_score > 0.5
        recommendations['heparin']['confidence'] = min(heparin_score, 1.0)
        
        # Combination therapy
        if recommendations['aspirin']['recommended'] and recommendations['heparin']['recommended']:
            recommendations['combination']['recommended'] = True
            recommendations['combination']['confidence'] = min(
                recommendations['aspirin']['confidence'], 
                recommendations['heparin']['confidence']
            )
            recommendations['combination']['reasoning'] = [
                'Both individual treatments recommended',
                'Synergistic effect expected'
            ]
        
        return recommendations


def create_treatment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create treatment features from a dataframe.
    """
    engineer = TreatmentFeatureEngineer()
    engineer.fit(df)
    return engineer.transform(df)


if __name__ == "__main__":
    # Example usage
    print("Treatment Feature Engineering Module")
    print("This module provides specialized feature engineering for IST treatment data")