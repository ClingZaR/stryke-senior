# Phase 2: Baseline Models Test Case Evaluation Summary

**Analysis Date:** 2025-09-12 13:24:57

## Test Cases Overview

**Case 1: Low Risk Young Adult**
- Description: Young, healthy individual with no risk factors
- Age: 25, Gender: Female
- Expected Risk Level: Low
- Key Risk Factors: None

**Case 2: Moderate Risk Middle-Aged**
- Description: Middle-aged with hypertension but otherwise healthy
- Age: 45, Gender: Male
- Expected Risk Level: Moderate
- Key Risk Factors: Hypertension, High Blood Pressure

**Case 3: High Risk Elderly**
- Description: Elderly with multiple risk factors
- Age: 75, Gender: Female
- Expected Risk Level: High
- Key Risk Factors: Hypertension, Heart Disease, High Blood Pressure

**Case 4: Diabetic Patient**
- Description: Middle-aged diabetic with elevated glucose
- Age: 55, Gender: Male
- Expected Risk Level: High
- Key Risk Factors: Hypertension, Diabetes, High Blood Pressure

**Case 5: Obese Patient**
- Description: Obese individual with elevated BMI
- Age: 50, Gender: Male
- Expected Risk Level: Moderate
- Key Risk Factors: Obesity

**Case 6: Heart Disease Patient**
- Description: Patient with existing heart disease
- Age: 62, Gender: Female
- Expected Risk Level: High
- Key Risk Factors: Hypertension, Heart Disease, High Blood Pressure

**Case 7: Elderly with Hypertension**
- Description: Elderly patient with hypertension
- Age: 68, Gender: Male
- Expected Risk Level: Moderate
- Key Risk Factors: Hypertension

**Case 8: Young Professional**
- Description: Young professional with minimal risk factors
- Age: 32, Gender: Female
- Expected Risk Level: Low
- Key Risk Factors: None

**Case 9: Multiple Comorbidities**
- Description: Patient with multiple health conditions
- Age: 70, Gender: Male
- Expected Risk Level: High
- Key Risk Factors: Hypertension, Heart Disease, Diabetes, Obesity, High Blood Pressure

**Case 10: Healthy Senior**
- Description: Healthy senior with minimal risk factors
- Age: 65, Gender: Female
- Expected Risk Level: Low
- Key Risk Factors: None

## Model Performance Summary

**Random Forest Standard:**
- Average Risk Probability: 0.590
- High Risk Cases: 6/10
- Low Risk Cases: 4/10

**Random Forest Balanced:**
- Average Risk Probability: 0.669
- High Risk Cases: 7/10
- Low Risk Cases: 3/10

**Logistic Regression Standard:**
- Average Risk Probability: 0.547
- High Risk Cases: 5/10
- Low Risk Cases: 5/10

**Logistic Regression Balanced:**
- Average Risk Probability: 0.597
- High Risk Cases: 6/10
- Low Risk Cases: 4/10

## Key Findings

- All baseline models successfully evaluated on diverse test cases
- Risk probabilities vary significantly across different patient profiles
- Model agreement analysis shows consistency patterns between algorithms
- Age, BMI, and comorbidities show strong correlation with risk predictions
- Balanced models tend to predict higher risk probabilities
- Random Forest models show different sensitivity patterns compared to Logistic Regression
