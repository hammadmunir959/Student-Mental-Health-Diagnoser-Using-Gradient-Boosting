#!/usr/bin/env python3
"""
Test script for the Streamlit app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_loading():
    """Test if models can be loaded successfully"""
    try:
        from api.predictor import MentalHealthPredictor
        from api.preprocessor import DataPreprocessor
        
        print("Testing model loading...")
        predictor = MentalHealthPredictor()
        preprocessor = DataPreprocessor()
        
        print("✅ Models loaded successfully!")
        
        # Test prediction with sample data
        sample_data = {
            'age': 22,
            'gender': 'Female',
            'academic_pressure': 4,
            'work_pressure': 3,
            'cgpa': 8.5,
            'study_satisfaction': 3,
            'job_satisfaction': 'Not Applicable',
            'work_study_hours': 6,
            'financial_stress': 2,
            'sleep_duration': '7-8 hours',
            'dietary_habits': 'Healthy',
            'suicidal_thoughts': 'No',
            'family_history': 'No',
            'city': 'Mumbai',
            'profession': 'Student',
            'degree': "Bachelor's"
        }
        
        print("Testing prediction with sample data...")
        processed_data = preprocessor.preprocess(sample_data)
        prediction_result = predictor.predict(processed_data)
        
        print(f"✅ Prediction successful!")
        print(f"   Prediction: {prediction_result['prediction']}")
        print(f"   Probability: {prediction_result['probability']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_streamlit_imports():
    """Test if all Streamlit dependencies are available"""
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib
        
        print("✅ All dependencies available!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧠 Testing Mental Health Diagnoser App")
    print("=" * 50)
    
    # Test dependencies
    print("\n1. Testing dependencies...")
    deps_ok = test_streamlit_imports()
    
    # Test model loading
    print("\n2. Testing model loading...")
    model_ok = test_model_loading()
    
    print("\n" + "=" * 50)
    if deps_ok and model_ok:
        print("🎉 All tests passed! The app is ready to run.")
        print("\nTo start the app, run:")
        print("streamlit run streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
