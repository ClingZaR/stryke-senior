# Model Comparison Summary Report

## Overview
This report summarizes the risk assessment results for 10 comprehensive test cases across 4 baseline models.

## Model Performance Statistics

### Random Forest (Standard)
- Mean Risk Score: 0.839
- Standard Deviation: 0.208
- Min Risk Score: 0.407
- Max Risk Score: 1.000
- Low Risk Cases (< 0.3): 0
- Moderate Risk Cases (0.3-0.7): 2
- High Risk Cases (≥ 0.7): 8

### Random Forest (Balanced)
- Mean Risk Score: 0.893
- Standard Deviation: 0.157
- Min Risk Score: 0.503
- Max Risk Score: 1.000
- Low Risk Cases (< 0.3): 0
- Moderate Risk Cases (0.3-0.7): 2
- High Risk Cases (≥ 0.7): 8

### Logistic Regression (Standard)
- Mean Risk Score: 0.854
- Standard Deviation: 0.184
- Min Risk Score: 0.457
- Max Risk Score: 1.000
- Low Risk Cases (< 0.3): 0
- Moderate Risk Cases (0.3-0.7): 2
- High Risk Cases (≥ 0.7): 8

### Logistic Regression (Balanced)
- Mean Risk Score: 0.924
- Standard Deviation: 0.106
- Min Risk Score: 0.629
- Max Risk Score: 1.000
- Low Risk Cases (< 0.3): 0
- Moderate Risk Cases (0.3-0.7): 1
- High Risk Cases (≥ 0.7): 9

## Individual Case Analysis

### Case 1: Low Risk Young Adult
- Random Forest (Standard): 0.407 (Moderate Risk)
- Random Forest (Balanced): 0.503 (Moderate Risk)
- Logistic Regression (Standard): 0.457 (Moderate Risk)
- Logistic Regression (Balanced): 0.629 (Moderate Risk)

### Case 2: Moderate Risk Middle-Aged
- Random Forest (Standard): 0.953 (High Risk)
- Random Forest (Balanced): 0.988 (High Risk)
- Logistic Regression (Standard): 1.000 (High Risk)
- Logistic Regression (Balanced): 1.000 (High Risk)

### Case 3: High Risk Elderly with Anticoagulation
- Random Forest (Standard): 0.977 (High Risk)
- Random Forest (Balanced): 1.000 (High Risk)
- Logistic Regression (Standard): 0.977 (High Risk)
- Logistic Regression (Balanced): 0.977 (High Risk)

### Case 4: Diabetic Patient with Poor Control
- Random Forest (Standard): 1.000 (High Risk)
- Random Forest (Balanced): 0.904 (High Risk)
- Logistic Regression (Standard): 0.914 (High Risk)
- Logistic Regression (Balanced): 0.972 (High Risk)

### Case 5: Obese Patient with Metabolic Syndrome
- Random Forest (Standard): 0.798 (High Risk)
- Random Forest (Balanced): 1.000 (High Risk)
- Logistic Regression (Standard): 0.898 (High Risk)
- Logistic Regression (Balanced): 0.929 (High Risk)

### Case 6: Heart Disease Patient with Elevated Troponin
- Random Forest (Standard): 1.000 (High Risk)
- Random Forest (Balanced): 0.989 (High Risk)
- Logistic Regression (Standard): 1.000 (High Risk)
- Logistic Regression (Balanced): 0.929 (High Risk)

### Case 7: Elderly with Previous TIA
- Random Forest (Standard): 0.973 (High Risk)
- Random Forest (Balanced): 1.000 (High Risk)
- Logistic Regression (Standard): 0.942 (High Risk)
- Logistic Regression (Balanced): 1.000 (High Risk)

### Case 8: Young Professional on Contraceptives
- Random Forest (Standard): 0.505 (Moderate Risk)
- Random Forest (Balanced): 0.699 (Moderate Risk)
- Logistic Regression (Standard): 0.565 (Moderate Risk)
- Logistic Regression (Balanced): 0.866 (High Risk)

### Case 9: Multiple Comorbidities with Anticoagulation
- Random Forest (Standard): 0.999 (High Risk)
- Random Forest (Balanced): 0.947 (High Risk)
- Logistic Regression (Standard): 1.000 (High Risk)
- Logistic Regression (Balanced): 0.939 (High Risk)

### Case 10: Healthy Senior with Family History
- Random Forest (Standard): 0.775 (High Risk)
- Random Forest (Balanced): 0.902 (High Risk)
- Logistic Regression (Standard): 0.784 (High Risk)
- Logistic Regression (Balanced): 1.000 (High Risk)
