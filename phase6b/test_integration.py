#!/usr/bin/env python3
"""
Phase 6b Integration Test Suite
Tests the complete IST database integration and treatment recommendations
"""

import sys
import os
import requests
import json
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from phase6b.ist_integration import ISTDataHarmonizer
from phase6b.treatment_features import TreatmentFeatureEngineer, TreatmentOutcomeAnalyzer
from phase6b.pipeline import Phase6bPipeline

class Phase6bIntegrationTest:
    def __init__(self):
        self.api_base = "http://127.0.0.1:5000"
        self.test_patient = {
            "age": 67,
            "gender": "male",
            "nationality": "middle east",
            "weight": 78.5,
            "height": 172,
            "bp_sys": 138,
            "bp_dia": 86,
            "glucose": 6.1,
            "hba1c": 6.8,
            "cholesterol": 5.4,
            "troponin": 14,
            "hypertension": 1,
            "diabetes": 1,
            "atrial_fibrillation": 1,
            "aspirin_administered": 1,
            "heparin_administered": 0,
            "treatment_combination": "aspirin_only",
            "treatment_timing": 2.5
        }
    
    def test_ist_data_harmonization(self):
        """Test IST data harmonization"""
        print("🧪 Testing IST Data Harmonization...")
        try:
            harmonizer = ISTDataHarmonizer()
            unified_data = harmonizer.harmonize_datasets()
            
            # Check if unified data has expected columns
            expected_cols = ['age', 'gender', 'aspirin_administered', 'heparin_administered', 'treatment_combination']
            missing_cols = [col for col in expected_cols if col not in unified_data.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return False
            
            print(f"✅ Data harmonization successful. Shape: {unified_data.shape}")
            print(f"   Columns: {list(unified_data.columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Data harmonization failed: {e}")
            return False
    
    def test_treatment_feature_engineering(self):
        """Test treatment feature engineering"""
        print("\n🧪 Testing Treatment Feature Engineering...")
        try:
            # Create sample data
            sample_data = pd.DataFrame([self.test_patient])
            
            engineer = TreatmentFeatureEngineer()
            enhanced_data = engineer.create_treatment_features(sample_data)
            
            # Check for new treatment features
            expected_features = ['early_treatment', 'delayed_treatment', 'age_aspirin_interaction']
            missing_features = [feat for feat in expected_features if feat not in enhanced_data.columns]
            
            if missing_features:
                print(f"❌ Missing treatment features: {missing_features}")
                return False
            
            print(f"✅ Treatment feature engineering successful")
            print(f"   New features: {[col for col in enhanced_data.columns if col not in sample_data.columns]}")
            return True
            
        except Exception as e:
            print(f"❌ Treatment feature engineering failed: {e}")
            return False
    
    def test_api_prediction(self):
        """Test API prediction endpoint"""
        print("\n🧪 Testing API Prediction Endpoint...")
        try:
            response = requests.post(
                f"{self.api_base}/predict",
                json=self.test_patient,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                print(f"❌ API prediction failed with status {response.status_code}")
                return False
            
            result = response.json()
            required_fields = ['probability', 'risk_level', 'explanation']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                print(f"❌ Missing response fields: {missing_fields}")
                return False
            
            print(f"✅ API prediction successful")
            print(f"   Probability: {result['probability']:.3f}")
            print(f"   Risk Level: {result['risk_level']}")
            return True
            
        except Exception as e:
            print(f"❌ API prediction failed: {e}")
            return False
    
    def test_treatment_recommendations(self):
        """Test treatment recommendations endpoint"""
        print("\n🧪 Testing Treatment Recommendations Endpoint...")
        try:
            response = requests.post(
                f"{self.api_base}/treatment-recommendations",
                json=self.test_patient,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                print(f"❌ Treatment recommendations failed with status {response.status_code}")
                return False
            
            result = response.json()
            required_treatments = ['aspirin', 'heparin', 'combination']
            
            for treatment in required_treatments:
                if treatment not in result:
                    print(f"❌ Missing treatment recommendation: {treatment}")
                    return False
                
                treatment_data = result[treatment]
                required_fields = ['recommended', 'confidence', 'reasoning']
                missing_fields = [field for field in required_fields if field not in treatment_data]
                
                if missing_fields:
                    print(f"❌ Missing fields in {treatment}: {missing_fields}")
                    return False
            
            print(f"✅ Treatment recommendations successful")
            for treatment in required_treatments:
                rec = result[treatment]
                status = "✓" if rec['recommended'] else "✗"
                print(f"   {treatment.title()}: {status} (confidence: {rec['confidence']:.2f})")
            return True
            
        except Exception as e:
            print(f"❌ Treatment recommendations failed: {e}")
            return False
    
    def test_phase6b_pipeline(self):
        """Test Phase 6b pipeline"""
        print("\n🧪 Testing Phase 6b Pipeline...")
        try:
            pipeline = Phase6bPipeline()
            
            # Test pipeline initialization
            if not hasattr(pipeline, 'harmonizer'):
                print("❌ Pipeline missing harmonizer")
                return False
            
            if not hasattr(pipeline, 'feature_engineer'):
                print("❌ Pipeline missing feature engineer")
                return False
            
            print("✅ Phase 6b pipeline initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Phase 6b pipeline failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 Starting Phase 6b Integration Tests\n")
        print("=" * 60)
        
        tests = [
            self.test_ist_data_harmonization,
            self.test_treatment_feature_engineering,
            self.test_api_prediction,
            self.test_treatment_recommendations,
            self.test_phase6b_pipeline
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All Phase 6b integration tests PASSED!")
            print("✅ IST database integration is working correctly")
            print("✅ Treatment recommendations are functional")
            print("✅ Dashboard enhancements are ready")
        else:
            print(f"⚠️  {total - passed} tests failed. Please review the issues above.")
        
        return passed == total

if __name__ == "__main__":
    tester = Phase6bIntegrationTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)