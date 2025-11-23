#!/usr/bin/env python3
"""
Phase 1: Comprehensive Exploratory Data Analysis
Advanced Senior Project - Mortality Risk Prediction

This script performs comprehensive data exploration and visualization
to understand the dataset structure, patterns, and relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
import os
from datetime import datetime

# Suppress warnings and set backend
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

class Phase1EDA:
    def __init__(self, data_path='../clean_data.csv'):
        self.data_path = data_path
        self.visuals_dir = 'visuals'
        
        # Create visuals directory
        os.makedirs(self.visuals_dir, exist_ok=True)
        
        print("=" * 60)
        print("PHASE 1: COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
    def load_data(self):
        """Load and perform initial data inspection"""
        try:
            print("\n1. LOADING DATASET...")
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Basic info
            print(f"\nDataset Overview:")
            print(f"- Shape: {self.df.shape}")
            print(f"- Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False
    
    def basic_analysis(self):
        """Perform basic statistical analysis"""
        print("\n2. BASIC STATISTICAL ANALYSIS...")
        
        # Check for target variable
        target_cols = ['death', 'mortality', 'outcome', 'target']
        self.target_col = None
        
        for col in target_cols:
            if col in self.df.columns:
                self.target_col = col
                break
        
        if self.target_col:
            print(f"✓ Target variable found: '{self.target_col}'")
            target_dist = self.df[self.target_col].value_counts()
            print(f"Target distribution:\n{target_dist}")
            
            # Calculate class imbalance ratio
            if len(target_dist) == 2:
                minority_class = target_dist.min()
                majority_class = target_dist.max()
                imbalance_ratio = majority_class / minority_class
                print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
        else:
            print("⚠ No clear target variable found")
            
        # Data types
        print(f"\nData Types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"- {dtype}: {count} columns")
    
    def missing_value_analysis(self):
        """Analyze missing values and create visualizations"""
        print("\n3. MISSING VALUE ANALYSIS...")
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Filter columns with missing values
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            print(f"✓ Found {len(missing_df)} columns with missing values")
            print(missing_df.head(10))
            
            # Create missing value visualization
            plt.figure(figsize=(12, 8))
            
            if len(missing_df) <= 20:  # Show all if reasonable number
                sns.barplot(data=missing_df.head(20), y=missing_df.head(20).index, x='Missing_Percentage')
                plt.title('Missing Values by Feature (Top 20)', fontsize=14, fontweight='bold')
                plt.xlabel('Missing Percentage (%)')
                plt.ylabel('Features')
                plt.tight_layout()
                plt.savefig(f'{self.visuals_dir}/missing_values_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Missing values visualization saved")
        else:
            print("✓ No missing values found in the dataset")
    
    def target_analysis(self):
        """Analyze target variable distribution"""
        if not self.target_col:
            print("\n4. TARGET ANALYSIS: Skipped (no target variable found)")
            return
            
        print("\n4. TARGET VARIABLE ANALYSIS...")
        
        # Create target distribution plot
        plt.figure(figsize=(10, 6))
        
        target_counts = self.df[self.target_col].value_counts()
        
        # Bar plot
        plt.subplot(1, 2, 1)
        target_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Target Variable Distribution', fontweight='bold')
        plt.xlabel(self.target_col.title())
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
                colors=['skyblue', 'salmon'], startangle=90)
        plt.title('Target Variable Proportion', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.visuals_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Target distribution visualization saved")
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("\n5. CORRELATION ANALYSIS...")
        
        # Select only numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("⚠ Insufficient numeric columns for correlation analysis")
            return
            
        print(f"✓ Analyzing correlations for {len(numeric_cols)} numeric features")
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.visuals_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Correlation matrix visualization saved")
        
        # Target correlations if available
        if self.target_col and self.target_col in numeric_cols:
            target_corr = corr_matrix[self.target_col].drop(self.target_col).sort_values(key=abs, ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_corr = target_corr.head(15)
            colors = ['red' if x < 0 else 'blue' for x in top_corr.values]
            top_corr.plot(kind='barh', color=colors)
            plt.title(f'Top 15 Features Correlated with {self.target_col.title()}', fontweight='bold')
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            plt.savefig(f'{self.visuals_dir}/target_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Target correlations visualization saved")
    
    def feature_distributions(self):
        """Analyze feature distributions"""
        print("\n6. FEATURE DISTRIBUTION ANALYSIS...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("⚠ No numeric columns found for distribution analysis")
            return
            
        # Select top features for visualization (max 12)
        cols_to_plot = list(numeric_cols[:12])
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(cols_to_plot):
            if i < len(axes):
                self.df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'{col}', fontweight='bold')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle('Feature Distributions (Top 12 Numeric Features)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.visuals_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature distributions visualization saved ({len(cols_to_plot)} features)")
    
    def generate_summary(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*60)
        print("PHASE 1 ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   • Total samples: {self.df.shape[0]:,}")
        print(f"   • Total features: {self.df.shape[1]:,}")
        print(f"   • Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if self.target_col:
            target_dist = self.df[self.target_col].value_counts()
            print(f"\n🎯 Target Variable ({self.target_col}):")
            for value, count in target_dist.items():
                percentage = (count / len(self.df)) * 100
                print(f"   • {value}: {count:,} ({percentage:.1f}%)")
        
        # Missing values summary
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            print(f"\n⚠️  Data Quality:")
            print(f"   • Total missing values: {missing_count:,}")
            print(f"   • Columns with missing data: {(self.df.isnull().sum() > 0).sum()}")
        else:
            print(f"\n✅ Data Quality: No missing values detected")
        
        # Data types
        print(f"\n📈 Feature Types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   • {dtype}: {count} features")
        
        print(f"\n📁 Generated Visualizations:")
        viz_files = [f for f in os.listdir(self.visuals_dir) if f.endswith('.png')]
        for viz_file in sorted(viz_files):
            print(f"   • {viz_file}")
        
        print(f"\n🎯 Key Insights for Next Phases:")
        if self.target_col and self.target_col in self.df.columns:
            target_dist = self.df[self.target_col].value_counts()
            if len(target_dist) == 2:
                imbalance_ratio = target_dist.max() / target_dist.min()
                if imbalance_ratio > 2:
                    print(f"   • Significant class imbalance detected ({imbalance_ratio:.1f}:1) - SMOTE recommended")
                else:
                    print(f"   • Balanced classes detected ({imbalance_ratio:.1f}:1)")
        
        missing_percent = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        if missing_percent > 5:
            print(f"   • High missing data rate ({missing_percent:.1f}%) - Advanced imputation needed")
        elif missing_percent > 0:
            print(f"   • Low missing data rate ({missing_percent:.1f}%) - Simple imputation sufficient")
        
        numeric_features = len(self.df.select_dtypes(include=[np.number]).columns)
        if numeric_features > 50:
            print(f"   • High-dimensional dataset ({numeric_features} numeric features) - Feature selection recommended")
        
        print(f"\n✅ Phase 1 Complete! Ready for Phase 2 (Baseline Models & Preprocessing)")
        print("="*60)
    
    def run_analysis(self):
        """Execute complete Phase 1 analysis"""
        if not self.load_data():
            return False
            
        self.basic_analysis()
        self.missing_value_analysis()
        self.target_analysis()
        self.correlation_analysis()
        self.feature_distributions()
        self.generate_summary()
        
        return True

if __name__ == "__main__":
    # Initialize and run Phase 1 EDA
    eda = Phase1EDA()
    success = eda.run_analysis()
    
    if success:
        print(f"\n🎉 Phase 1 analysis completed successfully!")
        print(f"📁 All visualizations saved to: {eda.visuals_dir}/")
    else:
        print(f"\n❌ Phase 1 analysis failed. Please check the data path and try again.")