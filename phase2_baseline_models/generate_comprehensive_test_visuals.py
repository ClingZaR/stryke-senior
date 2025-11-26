#!/usr/bin/env python3
"""
Comprehensive Test Case Visualization Generator
Generates detailed visualizations for the enhanced test cases with all clinical parameters
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveTestVisualizer:
    def __init__(self):
        self.test_cases_file = 'visuals/test_cases/comprehensive_test_case_parameters.json'
        self.output_dir = Path('visuals/test_cases/comprehensive_visuals')
        self.output_dir.mkdir(exist_ok=True)
        self.test_data = None
        self.df = None
        
    def load_test_data(self):
        """Load comprehensive test case data"""
        print("Loading comprehensive test case data...")
        
        with open(self.test_cases_file, 'r') as f:
            self.test_data = json.load(f)
        
        # Convert to DataFrame
        cases = []
        for case in self.test_data['test_cases']:
            case_data = case['parameters'].copy()
            case_data['case_id'] = case['case_id']
            case_data['name'] = case['name']
            case_data['description'] = case['description']
            cases.append(case_data)
        
        self.df = pd.DataFrame(cases)
        print(f"✓ Loaded {len(self.df)} comprehensive test cases")
        
    def create_risk_distribution_chart(self):
        """Create risk category distribution visualization"""
        print("Creating risk distribution chart...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk distribution pie chart
        risk_counts = self.df['expected_risk'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        
        wedges, texts, autotexts = ax1.pie(risk_counts.values, 
                                          labels=risk_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90,
                                          explode=(0.05, 0.05, 0.05))
        
        ax1.set_title('Risk Category Distribution\n(Comprehensive Test Cases)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Risk by age group
        self.df['age_group'] = pd.cut(self.df['Age'], 
                                     bins=[0, 35, 50, 65, 100], 
                                     labels=['Young (≤35)', 'Middle (36-50)', 
                                            'Senior (51-65)', 'Elderly (>65)'])
        
        risk_age_crosstab = pd.crosstab(self.df['age_group'], self.df['expected_risk'])
        risk_age_crosstab.plot(kind='bar', ax=ax2, color=colors, alpha=0.8)
        ax2.set_title('Risk Distribution by Age Group', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Age Group', fontsize=12)
        ax2.set_ylabel('Number of Cases', fontsize=12)
        ax2.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_risk_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Risk distribution chart saved")
        
    def create_clinical_markers_analysis(self):
        """Create comprehensive clinical markers analysis"""
        print("Creating clinical markers analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Key clinical markers to analyze
        markers = {
            'INR ': {'title': 'INR Levels', 'color': '#e74c3c', 'threshold': 2.0},
            'HbA1c (last one before admission)': {'title': 'HbA1c Levels (%)', 'color': '#3498db', 'threshold': 7.0},
            'Troponin level at admission ': {'title': 'Troponin Levels (ng/L)', 'color': '#9b59b6', 'threshold': 50.0},
            'Total cholesterol (last one before admission)': {'title': 'Total Cholesterol', 'color': '#f39c12', 'threshold': 5.0},
            'BMI': {'title': 'Body Mass Index', 'color': '#2ecc71', 'threshold': 25.0},
            'BP_sys': {'title': 'Systolic BP (mmHg)', 'color': '#e67e22', 'threshold': 140.0}
        }
        
        for i, (marker, config) in enumerate(markers.items()):
            ax = axes[i]
            
            # Create scatter plot with risk color coding
            risk_colors = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'}
            
            for risk in ['Low', 'Moderate', 'High']:
                mask = self.df['expected_risk'] == risk
                if mask.any():
                    ax.scatter(self.df.loc[mask, 'case_id'], 
                              self.df.loc[mask, marker],
                              c=risk_colors[risk], 
                              label=f'{risk} Risk',
                              s=100, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add threshold line if applicable
            if 'threshold' in config:
                ax.axhline(y=config['threshold'], color='red', linestyle='--', 
                          alpha=0.7, label=f'Threshold: {config["threshold"]}')
            
            ax.set_title(config['title'], fontsize=12, fontweight='bold')
            ax.set_xlabel('Case ID', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_clinical_markers.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Clinical markers analysis saved")
        
    def create_comorbidity_analysis(self):
        """Create comorbidity pattern analysis"""
        print("Creating comorbidity analysis...")
        
        # Define comorbidity columns
        comorbidity_cols = [
            'Known case of Hypertension (YES/NO)',
            'known case of diabetes (YES/NO)',
            'known case of coronary heart disease (YES/NO)',
            'known case of atrial fibrillation (YES/NO)'
        ]
        
        # Convert yes/no to 1/0
        comorbidity_data = self.df[comorbidity_cols + ['expected_risk', 'case_id', 'name']].copy()
        for col in comorbidity_cols:
            comorbidity_data[col] = (comorbidity_data[col] == 'yes').astype(int)
        
        # Calculate comorbidity count
        comorbidity_data['comorbidity_count'] = comorbidity_data[comorbidity_cols].sum(axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Comorbidity heatmap
        ax1 = axes[0, 0]
        heatmap_data = comorbidity_data[comorbidity_cols].T
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', 
                   xticklabels=[f"Case {i+1}" for i in range(len(comorbidity_data))],
                   yticklabels=[col.replace(' (YES/NO)', '') for col in comorbidity_cols],
                   ax=ax1, cbar_kws={'label': 'Present (1) / Absent (0)'})
        ax1.set_title('Comorbidity Pattern Heatmap', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Test Cases', fontsize=12)
        
        # Comorbidity count by risk
        ax2 = axes[0, 1]
        risk_colors = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'}
        for risk in ['Low', 'Moderate', 'High']:
            mask = comorbidity_data['expected_risk'] == risk
            if mask.any():
                ax2.scatter(comorbidity_data.loc[mask, 'case_id'], 
                           comorbidity_data.loc[mask, 'comorbidity_count'],
                           c=risk_colors[risk], label=f'{risk} Risk', s=120, alpha=0.7)
        
        ax2.set_title('Comorbidity Count by Risk Level', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Case ID', fontsize=12)
        ax2.set_ylabel('Number of Comorbidities', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.5, 4.5)
        
        # Individual comorbidity prevalence
        ax3 = axes[1, 0]
        comorbidity_prev = comorbidity_data[comorbidity_cols].mean()
        bars = ax3.bar(range(len(comorbidity_prev)), comorbidity_prev.values, 
                      color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'], alpha=0.8)
        ax3.set_title('Comorbidity Prevalence in Test Cases', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Comorbidity Type', fontsize=12)
        ax3.set_ylabel('Prevalence (Proportion)', fontsize=12)
        ax3.set_xticks(range(len(comorbidity_prev)))
        ax3.set_xticklabels([col.replace(' (YES/NO)', '').replace('known case of ', '') 
                            for col in comorbidity_cols], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, comorbidity_prev.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Risk distribution by comorbidity count
        ax4 = axes[1, 1]
        risk_by_comorbidity = pd.crosstab(comorbidity_data['comorbidity_count'], 
                                         comorbidity_data['expected_risk'])
        risk_by_comorbidity.plot(kind='bar', ax=ax4, 
                               color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
        ax4.set_title('Risk Level by Comorbidity Count', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Comorbidities', fontsize=12)
        ax4.set_ylabel('Number of Cases', fontsize=12)
        ax4.legend(title='Risk Level')
        ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_comorbidity_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Comorbidity analysis saved")
        
    def create_medication_risk_analysis(self):
        """Create medication and risk factor analysis"""
        print("Creating medication and risk analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Anticoagulation analysis
        ax1 = axes[0, 0]
        anticoag_data = self.df[['Are they on warfarin, or heparin before admission (YES/NO)', 
                                'INR ', 'expected_risk']].copy()
        anticoag_data['on_anticoag'] = (anticoag_data['Are they on warfarin, or heparin before admission (YES/NO)'] == 'yes')
        
        # INR levels by anticoagulation status
        for status in [True, False]:
            mask = anticoag_data['on_anticoag'] == status
            if mask.any():
                label = 'On Anticoagulation' if status else 'Not on Anticoagulation'
                ax1.scatter(anticoag_data.loc[mask].index, 
                           anticoag_data.loc[mask, 'INR '],
                           label=label, s=100, alpha=0.7)
        
        ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Therapeutic INR (2.0)')
        ax1.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7, label='High INR (3.0)')
        ax1.set_title('INR Levels by Anticoagulation Status', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Case Index', fontsize=12)
        ax1.set_ylabel('INR Level', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ECG findings distribution
        ax2 = axes[0, 1]
        ecg_counts = self.df['ECG at admission'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(ecg_counts)))
        bars = ax2.bar(range(len(ecg_counts)), ecg_counts.values, color=colors, alpha=0.8)
        ax2.set_title('ECG Findings Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('ECG Finding', fontsize=12)
        ax2.set_ylabel('Number of Cases', fontsize=12)
        ax2.set_xticks(range(len(ecg_counts)))
        ax2.set_xticklabels(ecg_counts.index, rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, ecg_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # Gender and contraceptive analysis
        ax3 = axes[1, 0]
        female_data = self.df[self.df['Gender'] == 'Female'].copy()
        if not female_data.empty:
            contraceptive_risk = pd.crosstab(female_data['oral contraceptive use in female(YES/NO)'], 
                                           female_data['expected_risk'])
            contraceptive_risk.plot(kind='bar', ax=ax3, 
                                  color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
            ax3.set_title('Risk by Contraceptive Use (Females)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Contraceptive Use', fontsize=12)
            ax3.set_ylabel('Number of Cases', fontsize=12)
            ax3.legend(title='Risk Level')
            ax3.tick_params(axis='x', rotation=45)
        
        # Family history impact
        ax4 = axes[1, 1]
        family_history_risk = pd.crosstab(self.df['Family History of Stroke '], 
                                        self.df['expected_risk'])
        family_history_risk.plot(kind='bar', ax=ax4, 
                               color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
        ax4.set_title('Risk by Family History of Stroke', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Family History', fontsize=12)
        ax4.set_ylabel('Number of Cases', fontsize=12)
        ax4.legend(title='Risk Level')
        ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_medication_risk_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Medication and risk analysis saved")
        
    def create_parameter_correlation_heatmap(self):
        """Create correlation heatmap of numerical parameters"""
        print("Creating parameter correlation heatmap...")
        
        # Select numerical columns
        numerical_cols = [
            'Age', 'Weight on admission', 'Height', 'BMI', 'BP_sys', 'BP_dia',
            'Blood glucose at admission ', 'HbA1c (last one before admission)',
            'INR ', 'Total cholesterol (last one before admission)',
            'Troponin level at admission '
        ]
        
        # Create correlation matrix
        corr_data = self.df[numerical_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax,
                   fmt='.2f', annot_kws={'size': 9})
        
        ax.set_title('Parameter Correlation Matrix\n(Comprehensive Test Cases)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_parameter_correlations.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Parameter correlation heatmap saved")
        
    def create_comprehensive_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        print("Creating comprehensive summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Comprehensive Test Cases Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Risk distribution pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        risk_counts = self.df['expected_risk'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax1.set_title('Risk Distribution', fontweight='bold')
        
        # 2. Age vs Risk scatter
        ax2 = fig.add_subplot(gs[0, 1])
        risk_colors = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'}
        for risk in ['Low', 'Moderate', 'High']:
            mask = self.df['expected_risk'] == risk
            if mask.any():
                ax2.scatter(self.df.loc[mask, 'Age'], 
                           self.df.loc[mask, 'BMI'],
                           c=risk_colors[risk], label=f'{risk} Risk', s=80, alpha=0.7)
        ax2.set_xlabel('Age')
        ax2.set_ylabel('BMI')
        ax2.set_title('Age vs BMI by Risk', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. INR levels
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(range(len(self.df)), self.df['INR '], 
               color=['#e74c3c' if x >= 2.0 else '#3498db' for x in self.df['INR ']], alpha=0.8)
        ax3.axhline(y=2.0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('INR Levels by Case', fontweight='bold')
        ax3.set_xlabel('Case ID')
        ax3.set_ylabel('INR')
        
        # 4. HbA1c levels
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.bar(range(len(self.df)), self.df['HbA1c (last one before admission)'], 
               color=['#e74c3c' if x >= 7.0 else '#2ecc71' for x in self.df['HbA1c (last one before admission)']], alpha=0.8)
        ax4.axhline(y=7.0, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('HbA1c Levels by Case', fontweight='bold')
        ax4.set_xlabel('Case ID')
        ax4.set_ylabel('HbA1c (%)')
        
        # 5. Comorbidity heatmap (simplified)
        ax5 = fig.add_subplot(gs[1, :])
        comorbidity_cols = [
            'Known case of Hypertension (YES/NO)',
            'known case of diabetes (YES/NO)',
            'known case of coronary heart disease (YES/NO)',
            'known case of atrial fibrillation (YES/NO)',
            'Personal previous history of stoke (YES/NO)',
            'Personal previous history of Transient Ischemic Attack (YES/NO)'
        ]
        
        comorbidity_matrix = self.df[comorbidity_cols].copy()
        for col in comorbidity_cols:
            comorbidity_matrix[col] = (comorbidity_matrix[col] == 'yes').astype(int)
        
        sns.heatmap(comorbidity_matrix.T, annot=True, cmap='RdYlBu_r', 
                   xticklabels=[f"Case {i+1}" for i in range(len(self.df))],
                   yticklabels=[col.replace(' (YES/NO)', '').replace('known case of ', '') for col in comorbidity_cols],
                   ax=ax5, cbar_kws={'label': 'Present (1) / Absent (0)'})
        ax5.set_title('Comorbidity Pattern Matrix', fontweight='bold')
        
        # 6. Blood pressure analysis
        ax6 = fig.add_subplot(gs[2, 0:2])
        for risk in ['Low', 'Moderate', 'High']:
            mask = self.df['expected_risk'] == risk
            if mask.any():
                ax6.scatter(self.df.loc[mask, 'BP_sys'], 
                           self.df.loc[mask, 'BP_dia'],
                           c=risk_colors[risk], label=f'{risk} Risk', s=100, alpha=0.7)
        ax6.axvline(x=140, color='red', linestyle='--', alpha=0.7, label='Hypertensive (140)')
        ax6.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Hypertensive (90)')
        ax6.set_xlabel('Systolic BP (mmHg)')
        ax6.set_ylabel('Diastolic BP (mmHg)')
        ax6.set_title('Blood Pressure Distribution by Risk', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Troponin levels
        ax7 = fig.add_subplot(gs[2, 2:4])
        troponin_data = self.df.sort_values('Troponin level at admission ')
        bars = ax7.bar(range(len(troponin_data)), troponin_data['Troponin level at admission '],
                      color=[risk_colors[risk] for risk in troponin_data['expected_risk']], alpha=0.8)
        ax7.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Elevated (50 ng/L)')
        ax7.set_title('Troponin Levels by Case (Sorted)', fontweight='bold')
        ax7.set_xlabel('Cases (Sorted by Troponin)')
        ax7.set_ylabel('Troponin (ng/L)')
        ax7.legend()
        
        # 8. Summary statistics table
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Create summary statistics
        summary_stats = {
            'Parameter': ['Age (years)', 'BMI', 'Systolic BP', 'Diastolic BP', 
                         'Glucose (mg/dL)', 'HbA1c (%)', 'INR', 'Cholesterol', 'Troponin (ng/L)'],
            'Mean': [f"{self.df['Age'].mean():.1f}",
                    f"{self.df['BMI'].mean():.1f}",
                    f"{self.df['BP_sys'].mean():.1f}",
                    f"{self.df['BP_dia'].mean():.1f}",
                    f"{self.df['Blood glucose at admission '].mean():.1f}",
                    f"{self.df['HbA1c (last one before admission)'].mean():.1f}",
                    f"{self.df['INR '].mean():.2f}",
                    f"{self.df['Total cholesterol (last one before admission)'].mean():.1f}",
                    f"{self.df['Troponin level at admission '].mean():.1f}"],
            'Min': [f"{self.df['Age'].min():.0f}",
                   f"{self.df['BMI'].min():.1f}",
                   f"{self.df['BP_sys'].min():.0f}",
                   f"{self.df['BP_dia'].min():.0f}",
                   f"{self.df['Blood glucose at admission '].min():.0f}",
                   f"{self.df['HbA1c (last one before admission)'].min():.1f}",
                   f"{self.df['INR '].min():.1f}",
                   f"{self.df['Total cholesterol (last one before admission)'].min():.1f}",
                   f"{self.df['Troponin level at admission '].min():.0f}"],
            'Max': [f"{self.df['Age'].max():.0f}",
                   f"{self.df['BMI'].max():.1f}",
                   f"{self.df['BP_sys'].max():.0f}",
                   f"{self.df['BP_dia'].max():.0f}",
                   f"{self.df['Blood glucose at admission '].max():.0f}",
                   f"{self.df['HbA1c (last one before admission)'].max():.1f}",
                   f"{self.df['INR '].max():.1f}",
                   f"{self.df['Total cholesterol (last one before admission)'].max():.1f}",
                   f"{self.df['Troponin level at admission '].max():.0f}"]
        }
        
        # Create table
        table_data = []
        for i in range(len(summary_stats['Parameter'])):
            table_data.append([summary_stats['Parameter'][i], 
                             summary_stats['Mean'][i],
                             summary_stats['Min'][i], 
                             summary_stats['Max'][i]])
        
        table = ax8.table(cellText=table_data,
                         colLabels=['Parameter', 'Mean', 'Min', 'Max'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_stats['Parameter']) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#3498db')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
        
        ax8.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.savefig(self.output_dir / 'comprehensive_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Comprehensive dashboard saved")
        
    def generate_all_visualizations(self):
        """Generate all comprehensive visualizations"""
        print("\n🎨 GENERATING COMPREHENSIVE TEST CASE VISUALIZATIONS")
        print("=" * 60)
        
        self.load_test_data()
        
        # Generate all visualizations
        self.create_risk_distribution_chart()
        self.create_clinical_markers_analysis()
        self.create_comorbidity_analysis()
        self.create_medication_risk_analysis()
        self.create_parameter_correlation_heatmap()
        self.create_comprehensive_summary_dashboard()
        
        print("\n✅ ALL COMPREHENSIVE VISUALIZATIONS GENERATED!")
        print(f"📁 Saved to: {self.output_dir}")
        print("\n📊 Generated visualizations:")
        print("   • comprehensive_risk_distribution.png")
        print("   • comprehensive_clinical_markers.png")
        print("   • comprehensive_comorbidity_analysis.png")
        print("   • comprehensive_medication_risk_analysis.png")
        print("   • comprehensive_parameter_correlations.png")
        print("   • comprehensive_dashboard.png")
        
if __name__ == "__main__":
    visualizer = ComprehensiveTestVisualizer()
    visualizer.generate_all_visualizations()