#!/usr/bin/env python3
"""
Comprehensive Individual Case Comparison Visualization
Generates detailed side-by-side comparisons of all test cases with comprehensive clinical parameters
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

class IndividualCaseComparator:
    def __init__(self):
        self.test_cases_file = 'visuals/test_cases/comprehensive_visuals/comprehensive_test_case_parameters.json'
        self.output_dir = Path('visuals/test_cases/comprehensive_visuals')
        self.output_dir.mkdir(exist_ok=True)
        self.test_data = None
        self.df = None
        
    def load_test_data(self):
        """Load comprehensive test case data"""
        print("Loading comprehensive test case data for individual comparison...")
        
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
        print(f"✓ Loaded {len(self.df)} comprehensive test cases for comparison")
        
    def create_individual_case_profiles(self):
        """Create detailed profile for each individual case"""
        print("Creating individual case profiles...")
        
        # Create individual detailed profiles for each case
        for i, (_, case) in enumerate(self.df.iterrows()):
            self.create_individual_case_profile(case, i+1)
        
        print(f"✓ Individual case profiles saved ({len(self.df)} profiles)")
    
    def create_individual_case_profile(self, case, case_number):
        """Create detailed profile for a single case"""
        
        # Define parameter categories
        demographics = ['Age', 'Gender', 'Weight on admission', 'Height', 'BMI']
        vital_signs = ['BP_sys', 'BP_dia', 'Blood glucose at admission ']
        lab_values = ['HbA1c (last one before admission)', 'INR ', 
                     'Total cholesterol (last one before admission)', 'Troponin level at admission ']
        comorbidities = ['Known case of Hypertension (YES/NO)', 'known case of diabetes (YES/NO)',
                        'known case of coronary heart disease (YES/NO)', 'known case of atrial fibrillation (YES/NO)',
                        'Personal previous history of stoke (YES/NO)', 
                        'Personal previous history of Transient Ischemic Attack (YES/NO)']
        medications = ['Are they on warfarin, or heparin before admission (YES/NO)', 
                      'oral contraceptive use in female(YES/NO)']
        other_factors = ['Family History of Stroke ', 'ECG at admission']
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'Comprehensive Case Profile: {case["name"]} (Case {case_number})', 
                    fontsize=18, fontweight='bold', y=0.98)
            
        # Demographics profile
        ax1 = axes[0, 0]
        demo_values = [case[col] if col in case and isinstance(case[col], (int, float)) else 0 for col in demographics]
        demo_labels = [col.replace(' on admission', '').replace(' (last one before admission)', '') for col in demographics]
        
        bars = ax1.bar(range(len(demographics)), demo_values, alpha=0.8, color='#3498db')
        ax1.set_title('Demographics Profile', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Values')
        ax1.set_xticks(range(len(demographics)))
        ax1.set_xticklabels(demo_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, demo_values):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(demo_values)*0.01, 
                        f'{value}', ha='center', va='bottom', fontweight='bold')
            
        # Vital Signs profile
        ax2 = axes[0, 1]
        vital_values = [case[col] if col in case else 0 for col in vital_signs]
        vital_labels = ['Systolic BP', 'Diastolic BP', 'Glucose']
        
        bars = ax2.bar(range(len(vital_signs)), vital_values, alpha=0.8, color='#2ecc71')
        ax2.set_title('Vital Signs Profile', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Values')
        ax2.set_xticks(range(len(vital_signs)))
        ax2.set_xticklabels(vital_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels and reference lines
        for bar, value in zip(bars, vital_values):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vital_values)*0.01, 
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # Add reference lines for normal ranges
        ax2.axhline(y=120, color='orange', linestyle='--', alpha=0.7, label='Normal SBP (120)')
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Normal DBP (80)')
        ax2.legend()
            
        # Lab Values profile
        ax3 = axes[0, 2]
        lab_values_list = [case[col] if col in case else 0 for col in lab_values]
        lab_labels = ['HbA1c', 'INR', 'Cholesterol', 'Troponin']
        
        bars = ax3.bar(range(len(lab_values)), lab_values_list, alpha=0.8, color='#9b59b6')
        ax3.set_title('Laboratory Values Profile', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Values')
        ax3.set_xticks(range(len(lab_values)))
        ax3.set_xticklabels(lab_labels, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels and reference lines
        for bar, value in zip(bars, lab_values_list):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lab_values_list)*0.01, 
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # Add reference lines for normal ranges
        thresholds = [7.0, 2.0, 200.0, 50.0]
        threshold_labels = ['HbA1c Target (7%)', 'INR High (2.0)', 'Cholesterol High (200)', 'Troponin High (50)']
        colors = ['orange', 'red', 'orange', 'red']
        
        for i, (threshold, label, color) in enumerate(zip(thresholds, threshold_labels, colors)):
            if i < len(lab_values_list) and lab_values_list[i] > 0:
                ax3.axhline(y=threshold, color=color, linestyle='--', alpha=0.7)
            
        # Comorbidities profile
        ax4 = axes[1, 0]
        comorbidity_values = [(1 if case[col] == 'yes' else 0) for col in comorbidities if col in case]
        comorbidity_labels = [col.replace(' (YES/NO)', '').replace('known case of ', '') for col in comorbidities]
        
        colors = ['#e74c3c' if val == 1 else '#95a5a6' for val in comorbidity_values]
        bars = ax4.bar(range(len(comorbidity_values)), comorbidity_values, alpha=0.8, color=colors)
        ax4.set_title('Comorbidities Profile', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Comorbidities')
        ax4.set_ylabel('Present (1) / Absent (0)')
        ax4.set_xticks(range(len(comorbidity_values)))
        ax4.set_xticklabels(comorbidity_labels, rotation=45, ha='right')
        ax4.set_ylim(0, 1.2)
        ax4.grid(True, alpha=0.3)
        
        # Add labels
        for bar, value in zip(bars, comorbidity_values):
            label = 'Present' if value == 1 else 'Absent'
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    label, ha='center', va='bottom', fontweight='bold', fontsize=10)
            
        # Risk Assessment Radar Chart
        ax5 = plt.subplot(3, 3, 5, projection='polar')
        
        # Define risk factors for radar chart
        risk_factors = ['Age Risk', 'BP Risk', 'Diabetes Risk', 'Cardiac Risk', 'Coagulation Risk']
        
        def calculate_risk_scores(case_row):
            age_risk = min(case_row['Age'] / 80, 1.0)  # Normalize age risk
            bp_risk = min(max(case_row['BP_sys'] - 120, 0) / 60, 1.0)  # BP above 120
            diabetes_risk = 1.0 if case_row['known case of diabetes (YES/NO)'] == 'yes' else 0.0
            cardiac_risk = 1.0 if case_row['known case of coronary heart disease (YES/NO)'] == 'yes' else 0.0
            coag_risk = min(max(case_row['INR '] - 1.0, 0) / 2.0, 1.0)  # INR above 1.0
            return [age_risk, bp_risk, diabetes_risk, cardiac_risk, coag_risk]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(risk_factors), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        risk_scores = calculate_risk_scores(case)
        risk_scores += risk_scores[:1]  # Complete the circle
        
        ax5.plot(angles, risk_scores, 'o-', linewidth=3, label=case['name'], color='#3498db')
        ax5.fill(angles, risk_scores, alpha=0.25, color='#3498db')
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(risk_factors)
        ax5.set_ylim(0, 1)
        ax5.set_title('Risk Profile Assessment', fontweight='bold', fontsize=14, pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
        # Clinical Summary Table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_data = [[
            case['name'],
            case['expected_risk'],
            f"{case['Age']} years",
            case['Gender'],
            f"{case['BP_sys']}/{case['BP_dia']} mmHg",
            f"{case['HbA1c (last one before admission)']}%",
            f"{case['INR ']}",
            'Yes' if case['known case of diabetes (YES/NO)'] == 'yes' else 'No'
        ]]
        
        table_headers = ['Parameter', 'Value']
        table_data = [
            ['Case Name', case['name']],
            ['Risk Level', case['expected_risk']],
            ['Age', f"{case['Age']} years"],
            ['Gender', case['Gender']],
            ['Blood Pressure', f"{case['BP_sys']}/{case['BP_dia']} mmHg"],
            ['HbA1c', f"{case['HbA1c (last one before admission)']}%"],
            ['INR', f"{case['INR ']}"],
            ['Diabetes', 'Yes' if case['known case of diabetes (YES/NO)'] == 'yes' else 'No']
        ]
        
        table = ax6.table(cellText=table_data,
                         colLabels=table_headers,
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Color code by risk level
        risk_colors = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'}
        risk_level = case['expected_risk']
        if risk_level in risk_colors:
            table[(2, 1)].set_facecolor(risk_colors[risk_level])
            table[(2, 1)].set_alpha(0.5)
        
        ax6.set_title('Clinical Summary', fontweight='bold', fontsize=14)
            
        # Medication and Risk Factors
        ax7 = axes[2, 0]
        med_risk_labels = ['Anticoagulation', 'Family History', 'Previous Stroke', 'Previous TIA']
        
        med_risk_values = [
            1 if case['Are they on warfarin, or heparin before admission (YES/NO)'] == 'yes' else 0,
            1 if case['Family History of Stroke '] == 'yes' else 0,
            1 if case['Personal previous history of stoke (YES/NO)'] == 'yes' else 0,
            1 if case['Personal previous history of Transient Ischemic Attack (YES/NO)'] == 'yes' else 0
        ]
        
        colors = ['#e74c3c' if val == 1 else '#95a5a6' for val in med_risk_values]
        bars = ax7.bar(range(len(med_risk_labels)), med_risk_values, alpha=0.8, color=colors)
        
        ax7.set_title('Medications & Risk Factors', fontweight='bold', fontsize=14)
        ax7.set_xlabel('Risk Factors')
        ax7.set_ylabel('Present (1) / Absent (0)')
        ax7.set_xticks(range(len(med_risk_labels)))
        ax7.set_xticklabels(med_risk_labels, rotation=45, ha='right')
        ax7.set_ylim(0, 1.2)
        ax7.grid(True, alpha=0.3)
        
        # Add labels
        for bar, value in zip(bars, med_risk_values):
            label = 'Present' if value == 1 else 'Absent'
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    label, ha='center', va='bottom', fontweight='bold', fontsize=10)
            
        # Biomarker Threshold Analysis
        ax8 = axes[2, 1]
        
        biomarkers = ['INR ', 'HbA1c (last one before admission)', 'Troponin level at admission ', 'BP_sys']
        thresholds = [2.0, 7.0, 50.0, 140.0]
        biomarker_labels = ['INR', 'HbA1c', 'Troponin', 'Systolic BP']
        
        threshold_values = []
        for biomarker, threshold in zip(biomarkers, thresholds):
            if biomarker in case:
                value = case[biomarker]
                threshold_values.append(1 if value >= threshold else 0)
            else:
                threshold_values.append(0)
        
        colors = ['#e74c3c' if val == 1 else '#2ecc71' for val in threshold_values]
        bars = ax8.bar(range(len(biomarker_labels)), threshold_values, alpha=0.8, color=colors)
        
        ax8.set_title('Biomarker Threshold Analysis', fontweight='bold', fontsize=14)
        ax8.set_xlabel('Biomarkers')
        ax8.set_ylabel('Above Threshold (1) / Below (0)')
        ax8.set_xticks(range(len(biomarker_labels)))
        ax8.set_xticklabels(biomarker_labels)
        ax8.set_ylim(0, 1.2)
        ax8.grid(True, alpha=0.3)
        
        # Add threshold values as text
        threshold_texts = ['≥2.0', '≥7.0%', '≥50', '≥140']
        for bar, value, threshold_text in zip(bars, threshold_values, threshold_texts):
            status = f'Above {threshold_text}' if value == 1 else f'Below {threshold_text}'
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    status, ha='center', va='bottom', fontweight='bold', fontsize=9, rotation=45)
            
        # Overall Risk Score
        ax9 = axes[2, 2]
        
        # Calculate comprehensive risk score
        score = 0
        
        # Age factor (0-30 points)
        score += min(case['Age'] * 0.375, 30)  # Max 30 for age 80
        
        # Comorbidity factors (10 points each)
        comorbidity_score = sum([10 for col in comorbidities if col in case and case[col] == 'yes'])
        score += comorbidity_score
        
        # Biomarker factors
        if case['INR '] >= 2.0: score += 15
        if case['HbA1c (last one before admission)'] >= 7.0: score += 15
        if case['Troponin level at admission '] >= 50.0: score += 20
        if case['BP_sys'] >= 140: score += 10
        
        # Risk level color
        color = '#27ae60' if score < 50 else '#f39c12' if score < 80 else '#e74c3c'
        risk_level = 'Low' if score < 50 else 'Moderate' if score < 80 else 'High'
        
        # Create gauge-like visualization
        bars = ax9.bar(['Risk Score'], [score], color=color, alpha=0.8, width=0.5)
        ax9.set_title('Comprehensive Risk Score', fontweight='bold', fontsize=14)
        ax9.set_ylabel('Risk Score')
        ax9.set_ylim(0, 120)
        
        # Add score label on bar
        ax9.text(0, score + 3, f'{score:.0f}\n({risk_level} Risk)', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add risk level lines
        ax9.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk (50)')
        ax9.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='High Risk (80)')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'individual_case_{case_number}_{case["name"].replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_all_cases_matrix_comparison(self):
        """Create a comprehensive matrix comparing all cases"""
        print("Creating comprehensive all-cases matrix comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Comprehensive All Cases Matrix Comparison', fontsize=18, fontweight='bold')
        
        # 1. Numerical Parameters Heatmap
        ax1 = axes[0, 0]
        numerical_params = ['Age', 'BMI', 'BP_sys', 'BP_dia', 'Blood glucose at admission ', 
                           'HbA1c (last one before admission)', 'INR ', 
                           'Total cholesterol (last one before admission)', 'Troponin level at admission ']
        
        numerical_data = self.df[numerical_params].T
        numerical_data.columns = [f"Case {i+1}" for i in range(len(self.df))]
        
        # Normalize data for better visualization
        numerical_data_norm = (numerical_data - numerical_data.min(axis=1).values.reshape(-1, 1)) / \
                             (numerical_data.max(axis=1).values.reshape(-1, 1) - numerical_data.min(axis=1).values.reshape(-1, 1))
        
        sns.heatmap(numerical_data_norm, annot=False, cmap='RdYlBu_r', ax=ax1, 
                   cbar_kws={'label': 'Normalized Value (0-1)'}, 
                   yticklabels=[param.replace(' (last one before admission)', '').replace(' at admission', '') for param in numerical_params])
        ax1.set_title('Normalized Numerical Parameters Matrix', fontweight='bold')
        ax1.set_xlabel('Test Cases')
        
        # 2. Comorbidities Matrix
        ax2 = axes[0, 1]
        comorbidity_cols = ['Known case of Hypertension (YES/NO)', 'known case of diabetes (YES/NO)',
                           'known case of coronary heart disease (YES/NO)', 'known case of atrial fibrillation (YES/NO)',
                           'Personal previous history of stoke (YES/NO)', 
                           'Personal previous history of Transient Ischemic Attack (YES/NO)']
        
        comorbidity_matrix = self.df[comorbidity_cols].copy()
        for col in comorbidity_cols:
            comorbidity_matrix[col] = (comorbidity_matrix[col] == 'yes').astype(int)
        
        comorbidity_matrix = comorbidity_matrix.T
        comorbidity_matrix.columns = [f"Case {i+1}" for i in range(len(self.df))]
        
        sns.heatmap(comorbidity_matrix, annot=True, cmap='RdYlBu_r', ax=ax2, 
                   cbar_kws={'label': 'Present (1) / Absent (0)'}, fmt='d',
                   yticklabels=[col.replace(' (YES/NO)', '').replace('known case of ', '') for col in comorbidity_cols])
        ax2.set_title('Comorbidities Matrix', fontweight='bold')
        ax2.set_xlabel('Test Cases')
        
        # 3. Risk Level and Scores Comparison
        ax3 = axes[1, 0]
        
        # Calculate comprehensive risk scores for all cases
        risk_scores = []
        for _, case in self.df.iterrows():
            score = 0
            score += min(case['Age'] * 0.375, 30)
            comorbidity_score = sum([10 for col in comorbidity_cols if col in case and case[col] == 'yes'])
            score += comorbidity_score
            if case['INR '] >= 2.0: score += 15
            if case['HbA1c (last one before admission)'] >= 7.0: score += 15
            if case['Troponin level at admission '] >= 50.0: score += 20
            if case['BP_sys'] >= 140: score += 10
            risk_scores.append(score)
        
        case_names = [f"Case {i+1}\n{case['name']}" for i, (_, case) in enumerate(self.df.iterrows())]
        risk_colors = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'}
        colors = [risk_colors[case['expected_risk']] for _, case in self.df.iterrows()]
        
        bars = ax3.bar(range(len(self.df)), risk_scores, color=colors, alpha=0.8)
        ax3.set_title('Comprehensive Risk Scores by Case', fontweight='bold')
        ax3.set_xlabel('Test Cases')
        ax3.set_ylabel('Risk Score')
        ax3.set_xticks(range(len(self.df)))
        ax3.set_xticklabels([f"Case {i+1}" for i in range(len(self.df))], rotation=45)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, risk_scores)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{score:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add risk level lines
        ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk (50)')
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='High Risk (80)')
        ax3.legend()
        
        # 4. Biomarker Threshold Matrix
        ax4 = axes[1, 1]
        
        biomarkers = ['INR ', 'HbA1c (last one before admission)', 'Troponin level at admission ', 'BP_sys']
        thresholds = [2.0, 7.0, 50.0, 140.0]
        biomarker_labels = ['INR ≥2.0', 'HbA1c ≥7.0%', 'Troponin ≥50', 'SBP ≥140']
        
        threshold_matrix = []
        for biomarker, threshold in zip(biomarkers, thresholds):
            threshold_row = [(1 if case[biomarker] >= threshold else 0) for _, case in self.df.iterrows()]
            threshold_matrix.append(threshold_row)
        
        threshold_df = pd.DataFrame(threshold_matrix, 
                                  index=biomarker_labels,
                                  columns=[f"Case {i+1}" for i in range(len(self.df))])
        
        sns.heatmap(threshold_df, annot=True, cmap='RdYlBu_r', ax=ax4, 
                   cbar_kws={'label': 'Above Threshold (1) / Below (0)'}, fmt='d')
        ax4.set_title('Biomarker Threshold Matrix', fontweight='bold')
        ax4.set_xlabel('Test Cases')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_cases_matrix_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ All cases matrix comparison saved")
        
    def create_case_similarity_analysis(self):
        """Create similarity analysis between all cases"""
        print("Creating case similarity analysis...")
        
        # Prepare numerical data for similarity calculation
        numerical_cols = ['Age', 'BMI', 'BP_sys', 'BP_dia', 'Blood glucose at admission ', 
                         'HbA1c (last one before admission)', 'INR ', 
                         'Total cholesterol (last one before admission)', 'Troponin level at admission ']
        
        # Normalize numerical data
        numerical_data = self.df[numerical_cols].copy()
        numerical_data_norm = (numerical_data - numerical_data.min()) / (numerical_data.max() - numerical_data.min())
        
        # Add binary comorbidity data
        comorbidity_cols = ['Known case of Hypertension (YES/NO)', 'known case of diabetes (YES/NO)',
                           'known case of coronary heart disease (YES/NO)', 'known case of atrial fibrillation (YES/NO)']
        
        for col in comorbidity_cols:
            numerical_data_norm[col] = (self.df[col] == 'yes').astype(int)
        
        # Calculate similarity matrix (using correlation)
        similarity_matrix = numerical_data_norm.T.corr()
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Similarity heatmap
        ax1 = axes[0]
        case_labels = [f"Case {i+1}\n{case['name']}" for i, (_, case) in enumerate(self.df.iterrows())]
        
        sns.heatmap(similarity_matrix, annot=True, cmap='RdYlBu_r', ax=ax1, 
                   xticklabels=[f"Case {i+1}" for i in range(len(self.df))],
                   yticklabels=[f"Case {i+1}" for i in range(len(self.df))],
                   cbar_kws={'label': 'Similarity Score'}, fmt='.2f')
        ax1.set_title('Case Similarity Matrix\n(Based on Clinical Parameters)', fontweight='bold', fontsize=14)
        
        # Most similar pairs
        ax2 = axes[1]
        
        # Find most similar pairs (excluding diagonal)
        similarity_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                similarity_pairs.append((i, j, similarity_matrix.iloc[i, j]))
        
        # Sort by similarity score
        similarity_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = similarity_pairs[:5]  # Top 5 most similar pairs
        
        pair_labels = [f"Case {pair[0]+1} vs Case {pair[1]+1}" for pair in top_pairs]
        similarity_scores = [pair[2] for pair in top_pairs]
        
        bars = ax2.barh(range(len(top_pairs)), similarity_scores, 
                       color=['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c'], alpha=0.8)
        ax2.set_title('Top 5 Most Similar Case Pairs', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Similarity Score')
        ax2.set_yticks(range(len(top_pairs)))
        ax2.set_yticklabels(pair_labels)
        ax2.set_xlim(0, 1)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, similarity_scores)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'case_similarity_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Case similarity analysis saved")
        
    def generate_all_individual_comparisons(self):
        """Generate all individual case comparison visualizations"""
        print("\n🔍 GENERATING COMPREHENSIVE INDIVIDUAL CASE COMPARISONS")
        print("=" * 70)
        
        self.load_test_data()
        
        # Generate all comparison visualizations
        self.create_individual_case_profiles()
        self.create_all_cases_matrix_comparison()
        self.create_case_similarity_analysis()
        
        print("\n✅ ALL INDIVIDUAL CASE COMPARISONS GENERATED!")
        print(f"📁 All individual case comparison visualizations saved to: {self.output_dir}")
        print("\n📊 Generated visualizations:")
        print(f"   • individual_case_1_*.png to individual_case_{len(self.df)}_*.png (Individual case profiles)")
        print("   • all_cases_matrix_comparison.png (Complete matrix view)")
        print("   • case_similarity_analysis.png (Similarity analysis)")
        
if __name__ == "__main__":
    comparator = IndividualCaseComparator()
    comparator.generate_all_individual_comparisons()