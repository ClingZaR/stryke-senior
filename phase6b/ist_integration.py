"""
IST Database Integration Utilities for Phase 6b

This module provides functionality to integrate the International Stroke Trial (IST) 
database with the existing clinical dataset, including data harmonization, 
feature alignment, and quality assurance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class ISTDataHarmonizer:
    """
    Harmonizes IST database with clinical dataset for integrated modeling.
    """
    
    def __init__(self, clinical_data_path: str = '../clean_data.csv', 
                 ist_data_path: str = '../IST_data.csv'):
        self.clinical_data_path = clinical_data_path
        self.ist_data_path = ist_data_path
        self.feature_mapping = self._define_feature_mapping()
        self.label_encoders = {}
        
    def _define_feature_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Define mapping between IST and clinical dataset features.
        """
        return {
            'common_features': {
                # IST feature -> Clinical feature (using real IST column names)
                'AGE': 'Age',
                'SEX': 'Gender', 
                'RATRIAL': 'known case of atrial fibrillation (YES/NO)',
                'RSBP': 'BP_sys',
                'RXASP': 'Aspirin administered (YES/NO)'
            },
            'ist_specific': {
                # IST-specific treatment features (using real IST column names)
                'RXHEP': 'heparin_treatment',
                'RDELAY': 'treatment_timing_days',
                'HOSPNUM': 'patient_id'
            },
            'clinical_specific': {
                # Clinical dataset specific features
                'diabetes': 'known case of diabetes (YES/NO)',
                'hypertension': 'Known case of Hypertension (YES/NO)',
                'troponin': 'Troponin level at admission',
                'glucose': 'Blood glucose at admission',
                'cholesterol': 'Total cholesterol (last one before admission)',
                'bmi': 'BMI'
            }
        }
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both clinical and IST datasets.
        """
        try:
            clinical_df = pd.read_csv(self.clinical_data_path)
            print(f"Loaded clinical dataset: {clinical_df.shape}")
        except FileNotFoundError:
            print(f"Clinical dataset not found at {self.clinical_data_path}")
            clinical_df = pd.DataFrame()
            
        try:
            ist_df = pd.read_csv(self.ist_data_path)
            print(f"Loaded IST dataset: {ist_df.shape}")
        except FileNotFoundError:
            print(f"IST dataset not found at {self.ist_data_path}")
            print("Creating simulated IST data for demonstration...")
            ist_df = self._create_simulated_ist_data()
            
        return clinical_df, ist_df
    
    def _create_simulated_ist_data(self) -> pd.DataFrame:
        """
        Create simulated IST data for demonstration purposes.
        This simulates the key IST features based on the actual trial design.
        """
        np.random.seed(42)
        n_samples = 1000  # Smaller sample for demonstration
        
        # Simulate IST data structure
        ist_data = {
            'patient_id': range(1, n_samples + 1),
            'age': np.random.normal(70, 12, n_samples).clip(18, 95),
            'sex': np.random.choice(['male', 'female'], n_samples, p=[0.52, 0.48]),
            'atrial_fibrillation': np.random.choice(['yes', 'no'], n_samples, p=[0.15, 0.85]),
            'systolic_bp': np.random.normal(150, 25, n_samples).clip(90, 220),
            'diastolic_bp': np.random.normal(85, 15, n_samples).clip(50, 120),
            
            # Treatment variables (key IST features)
            'aspirin_administered': np.random.choice(['yes', 'no'], n_samples, p=[0.5, 0.5]),
            'heparin_administered': np.random.choice(['yes', 'no'], n_samples, p=[0.5, 0.5]),
            'treatment_timing_days': np.random.uniform(0, 3, n_samples),
            
            # Outcomes
            'death_outcome': np.random.choice(['yes', 'no'], n_samples, p=[0.12, 0.88]),
            'stroke_recurrence': np.random.choice(['yes', 'no'], n_samples, p=[0.08, 0.92])
        }
        
        # Create treatment combinations
        df = pd.DataFrame(ist_data)
        df['treatment_combination'] = df.apply(self._determine_treatment_combo, axis=1)
        df['randomization_arm'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        
        print(f"Created simulated IST dataset with {n_samples} samples")
        return df
    
    def _determine_treatment_combo(self, row) -> str:
        """Determine treatment combination based on individual treatments."""
        aspirin = row['aspirin_administered'] == 'yes'
        heparin = row['heparin_administered'] == 'yes'
        
        if aspirin and heparin:
            return 'aspirin_heparin'
        elif aspirin:
            return 'aspirin_only'
        elif heparin:
            return 'heparin_only'
        else:
            return 'neither'
    
    def harmonize_features(self, clinical_df: pd.DataFrame, 
                          ist_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Harmonize features between clinical and IST datasets.
        """
        # Standardize column names
        clinical_df.columns = [str(c).strip() for c in clinical_df.columns]
        ist_df.columns = [str(c).strip() for c in ist_df.columns]
        
        # Create harmonized clinical dataset
        clinical_harmonized = self._harmonize_clinical_data(clinical_df)
        
        # Create harmonized IST dataset  
        ist_harmonized = self._harmonize_ist_data(ist_df)
        
        return clinical_harmonized, ist_harmonized
    
    def _harmonize_clinical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize clinical dataset to common schema."""
        harmonized = df.copy()
        
        # Standardize yes/no columns
        yes_no_cols = [col for col in harmonized.columns if '(YES/NO)' in col]
        for col in yes_no_cols:
            harmonized[col] = harmonized[col].astype(str).str.lower()
            harmonized[col] = harmonized[col].map({'yes': 1, 'no': 0, 'nan': np.nan})
        
        # Add dataset source
        harmonized['data_source'] = 'clinical'
        
        # Add missing IST-specific features with NaN
        ist_features = ['heparin_treatment', 'treatment_combo', 'treatment_timing_days']
        for feature in ist_features:
            if feature not in harmonized.columns:
                harmonized[feature] = np.nan
                
        return harmonized
    
    def _harmonize_ist_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize IST dataset to common schema."""
        harmonized = df.copy()
        
        # Standardize yes/no columns
        yes_no_cols = ['aspirin_administered', 'heparin_administered', 'atrial_fibrillation']
        for col in yes_no_cols:
            if col in harmonized.columns:
                harmonized[col] = harmonized[col].astype(str).str.lower()
                harmonized[col] = harmonized[col].map({'yes': 1, 'no': 0})
        
        # Rename columns to match clinical dataset
        column_mapping = {
            # Real IST column names -> Harmonized names
            'SEX': 'Gender',
            'AGE': 'Age',
            'RSBP': 'BP_sys',
            'RXASP': 'Aspirin administered (YES/NO)',
            'RATRIAL': 'known case of atrial fibrillation (YES/NO)',
            'RXHEP': 'heparin_treatment',
            'RDELAY': 'treatment_timing_days',
            'HOSPNUM': 'patient_id',
            'DIED': 'Death outcome (YES/NO)'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in harmonized.columns:
                harmonized[new_col] = harmonized[old_col]
        
        # Convert IST-specific values to match clinical dataset format
        if 'Gender' in harmonized.columns:
            # Convert M/F to standard format
            harmonized['Gender'] = harmonized['Gender'].map({'M': 'Male', 'F': 'Female'})
        
        if 'Aspirin administered (YES/NO)' in harmonized.columns:
            # Convert Y/N to YES/NO format
            harmonized['Aspirin administered (YES/NO)'] = harmonized['Aspirin administered (YES/NO)'].map({'Y': 'YES', 'N': 'NO'})
        
        if 'heparin_treatment' in harmonized.columns:
            # Convert heparin codes to binary (any heparin treatment vs none)
            harmonized['heparin_treatment'] = harmonized['heparin_treatment'].apply(
                lambda x: 'YES' if x in ['H', 'M', 'L'] else 'NO'
            )
        
        if 'known case of atrial fibrillation (YES/NO)' in harmonized.columns:
            # Handle atrial fibrillation (mostly NaN in IST data)
            harmonized['known case of atrial fibrillation (YES/NO)'] = harmonized['known case of atrial fibrillation (YES/NO)'].fillna('NO')
        
        # Create treatment combination feature
        if all(col in harmonized.columns for col in ['Aspirin administered (YES/NO)', 'heparin_treatment']):
            def get_treatment_combo(row):
                aspirin = row['Aspirin administered (YES/NO)'] == 'YES'
                heparin = row['heparin_treatment'] == 'YES'
                if aspirin and heparin:
                    return 'aspirin_heparin'
                elif aspirin:
                    return 'aspirin_only'
                elif heparin:
                    return 'heparin_only'
                else:
                    return 'neither'
            
            harmonized['treatment_combo'] = harmonized.apply(get_treatment_combo, axis=1)
        
        # Add dataset source
        harmonized['data_source'] = 'ist'
        
        # Add missing clinical-specific features with NaN
        clinical_features = [
            'known case of diabetes (YES/NO)',
            'Known case of Hypertension (YES/NO)', 
            'Troponin level at admission',
            'Blood glucose at admission',
            'Total cholesterol (last one before admission)',
            'BMI'
        ]
        for feature in clinical_features:
            if feature not in harmonized.columns:
                harmonized[feature] = np.nan
                
        return harmonized
    
    def create_unified_dataset(self, clinical_df: pd.DataFrame, 
                              ist_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create unified dataset combining both sources.
        """
        # Harmonize features
        clinical_harm, ist_harm = self.harmonize_features(clinical_df, ist_df)
        
        # Handle case where one dataset is empty
        if clinical_harm.empty and not ist_harm.empty:
            unified_df = ist_harm.copy()
        elif ist_harm.empty and not clinical_harm.empty:
            unified_df = clinical_harm.copy()
        elif not clinical_harm.empty and not ist_harm.empty:
            # Find common columns
            common_cols = list(set(clinical_harm.columns) & set(ist_harm.columns))
            
            # Combine datasets
            clinical_subset = clinical_harm[common_cols]
            ist_subset = ist_harm[common_cols]
            
            unified_df = pd.concat([clinical_subset, ist_subset], ignore_index=True)
        else:
            # Both datasets are empty
            unified_df = pd.DataFrame()
        
        if not unified_df.empty:
            print(f"Created unified dataset: {unified_df.shape}")
            print(f"Data sources: {unified_df['data_source'].value_counts().to_dict()}")
        
        return unified_df
    
    def integrate_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Main integration function that returns all dataset variants.
        """
        # Load datasets
        clinical_df, ist_df = self.load_datasets()
        
        if clinical_df.empty or ist_df.empty:
            print("Warning: One or both datasets are empty")
            # If we have simulated IST data, still proceed with integration
            if not ist_df.empty:
                # Create unified dataset with simulated data
                unified_df = self.create_unified_dataset(clinical_df, ist_df)
                # Create treatment-enhanced features
                unified_enhanced = self._create_treatment_features(unified_df)
                
                return {
                    'clinical_original': clinical_df,
                    'ist_original': ist_df,
                    'unified': unified_df,
                    'enhanced': unified_enhanced
                }
            return {}
        
        # Create unified dataset
        unified_df = self.create_unified_dataset(clinical_df, ist_df)
        
        # Create treatment-enhanced features
        unified_enhanced = self._create_treatment_features(unified_df)
        
        return {
            'clinical_original': clinical_df,
            'ist_original': ist_df,
            'unified': unified_df,
            'enhanced': unified_enhanced
        }
    
    def _create_treatment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced treatment-related features.
        """
        enhanced_df = df.copy()
        
        # Create standardized column names for testing
        if 'Age' in enhanced_df.columns:
            enhanced_df['age'] = enhanced_df['Age']
        if 'Gender' in enhanced_df.columns:
            enhanced_df['gender'] = enhanced_df['Gender']
        if 'Aspirin administered (YES/NO)' in enhanced_df.columns:
            enhanced_df['aspirin_administered'] = enhanced_df['Aspirin administered (YES/NO)']
        if 'heparin_treatment' in enhanced_df.columns:
            enhanced_df['heparin_administered'] = enhanced_df['heparin_treatment']
        if 'treatment_combo' in enhanced_df.columns:
            enhanced_df['treatment_combination'] = enhanced_df['treatment_combo']
        
        # Treatment combination features
        if 'treatment_combo' in enhanced_df.columns:
            # One-hot encode treatment combinations
            treatment_dummies = pd.get_dummies(enhanced_df['treatment_combo'], 
                                             prefix='treatment', dummy_na=True)
            enhanced_df = pd.concat([enhanced_df, treatment_dummies], axis=1)
        
        # Treatment timing features
        if 'treatment_timing_days' in enhanced_df.columns:
            enhanced_df['early_treatment'] = (enhanced_df['treatment_timing_days'] <= 1).astype(int)
            enhanced_df['delayed_treatment'] = (enhanced_df['treatment_timing_days'] > 2).astype(int)
        
        # Interaction features
        if all(col in enhanced_df.columns for col in ['Age', 'Aspirin administered (YES/NO)']):
            enhanced_df['age_aspirin_interaction'] = (
                enhanced_df['Age'] * enhanced_df['Aspirin administered (YES/NO)']
            )
        
        return enhanced_df

    def harmonize_datasets(self) -> pd.DataFrame:
        """
        Alias for integrate_datasets method that returns the enhanced unified dataset.
        """
        datasets = self.integrate_datasets()
        return datasets.get('enhanced', pd.DataFrame()) if datasets else pd.DataFrame()


def validate_integration(datasets: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """
    Validate the integration results and provide quality metrics.
    """
    validation_results = {}
    
    if 'enhanced' in datasets:
        df = datasets['enhanced']
        
        validation_results.update({
            'total_samples': len(df),
            'data_source_distribution': df['data_source'].value_counts().to_dict(),
            'missing_value_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'treatment_feature_coverage': {
                'aspirin_coverage': df['Aspirin administered (YES/NO)'].notna().sum() / len(df),
                'heparin_coverage': df['heparin_treatment'].notna().sum() / len(df) if 'heparin_treatment' in df.columns else 0,
                'treatment_combo_coverage': df['treatment_combo'].notna().sum() / len(df) if 'treatment_combo' in df.columns else 0
            }
        })
    
    return validation_results


if __name__ == "__main__":
    # Example usage
    harmonizer = ISTDataHarmonizer()
    datasets = harmonizer.integrate_datasets()
    
    if datasets:
        validation = validate_integration(datasets)
        print("\nIntegration Validation Results:")
        for key, value in validation.items():
            print(f"{key}: {value}")