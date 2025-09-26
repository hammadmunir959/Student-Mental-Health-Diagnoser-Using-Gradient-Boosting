#!/usr/bin/env python3
"""
Mental Health Predictor Module
Handles model loading and prediction logic
"""

import joblib
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, Any, List
import json

logger = logging.getLogger(__name__)

class MentalHealthPredictor:
    """Mental Health Depression Predictor"""
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the predictor with model files
        
        Args:
            models_dir: Path to the models directory
        """
        if models_dir is None:
            # Default to the models directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(os.path.dirname(current_dir), 'models')
        
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.model_metadata = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all required model files"""
        try:
            # Load the main model (try different possible names)
            model_files = [
                'mental_health_model_20250926_165109.pkl',
                'logistic_regression_model.pkl',
                'logistic_regression_optimized.pkl'
            ]
            
            model_loaded = False
            for model_file in model_files:
                model_path = os.path.join(self.models_dir, model_file)
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                    logger.info(f"Loaded model from {model_file}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                raise FileNotFoundError("No model file found")
            
            # Load scaler
            scaler_files = [
                'scaler_20250926_165109.pkl',
                'scaler_optimized.pkl',
                'scaler.pkl'
            ]
            
            scaler_loaded = False
            for scaler_file in scaler_files:
                scaler_path = os.path.join(self.models_dir, scaler_file)
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler from {scaler_file}")
                    scaler_loaded = True
                    break
            
            if not scaler_loaded:
                raise FileNotFoundError("No scaler file found")
            
            # Load label encoders
            encoders_path = os.path.join(self.models_dir, 'label_encoders_20250926_165109.pkl')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                logger.info("Loaded label encoders")
            else:
                # Try to load categorical mappings as fallback
                mappings_path = os.path.join(self.models_dir, 'categorical_mappings.pkl')
                if os.path.exists(mappings_path):
                    self.label_encoders = joblib.load(mappings_path)
                    logger.info("Loaded categorical mappings as label encoders")
                else:
                    logger.warning("No label encoders found, will use default mappings")
            
            # Load feature columns
            feature_files = [
                'feature_columns_20250926_165109.pkl',
                'feature_columns_optimized.csv',
                'feature_columns.csv'
            ]
            
            feature_loaded = False
            for feature_file in feature_files:
                feature_path = os.path.join(self.models_dir, feature_file)
                if os.path.exists(feature_path):
                    if feature_file.endswith('.pkl'):
                        self.feature_columns = joblib.load(feature_path)
                    else:
                        self.feature_columns = pd.read_csv(feature_path)['feature'].tolist()
                    logger.info(f"Loaded feature columns from {feature_file}")
                    feature_loaded = True
                    break
            
            if not feature_loaded:
                # Use default feature columns based on the model metadata
                self.feature_columns = [
                    'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                    'Job Satisfaction', 'Work/Study Hours', 'Financial Stress', 'Sleep_Hours',
                    'Diet_Score', 'Risk_Score', 'Gender_encoded', 'Profession_encoded',
                    'Have you ever had suicidal thoughts ?_encoded', 'Family History of Mental Illness_encoded'
                ]
                logger.warning("Using default feature columns")
            
            # Load model metadata
            metadata_path = os.path.join(self.models_dir, 'model_metadata_20250926_165109.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("Loaded model metadata")
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise e
    
    def predict(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction using the loaded model
        
        Args:
            processed_data: Preprocessed input data
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if self.model is None or self.scaler is None:
                raise ValueError("Model or scaler not loaded")
            
            # Create feature vector in the correct order
            feature_vector = []
            for feature in self.feature_columns:
                if feature in processed_data:
                    feature_vector.append(processed_data[feature])
                else:
                    logger.warning(f"Feature {feature} not found in processed data")
                    feature_vector.append(0.0)  # Default value
            
            # Convert to DataFrame with proper column names
            features_df = pd.DataFrame([feature_vector], columns=self.feature_columns)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of positive class
            
            return {
                'prediction': int(prediction),
                'probability': float(probability)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'encoders_loaded': self.label_encoders is not None,
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns) if self.feature_columns else 0
        }
        
        if self.model_metadata:
            info.update({
                'model_name': self.model_metadata.get('model_name', 'Unknown'),
                'accuracy': self.model_metadata.get('accuracy', 'Unknown'),
                'auc': self.model_metadata.get('auc', 'Unknown'),
                'precision': self.model_metadata.get('precision', 'Unknown'),
                'recall': self.model_metadata.get('recall', 'Unknown'),
                'f1_score': self.model_metadata.get('f1_score', 'Unknown'),
                'training_samples': self.model_metadata.get('training_samples', 'Unknown'),
                'test_samples': self.model_metadata.get('test_samples', 'Unknown')
            })
        
        return info
