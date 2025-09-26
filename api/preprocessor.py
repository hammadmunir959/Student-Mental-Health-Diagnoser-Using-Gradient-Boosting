#!/usr/bin/env python3
"""
Data Preprocessor Module
Handles data validation, normalization, and feature engineering
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List
from api.predictor import MentalHealthPredictor

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data Preprocessor for Mental Health Diagnosis"""
    
    def __init__(self):
        """Initialize the preprocessor with default mappings"""
        # Sleep duration mapping
        self.sleep_mapping = {
            'Less than 5 hours': 4,
            '5-6 hours': 5.5,
            '7-8 hours': 7.5,
            'More than 8 hours': 9,
            'Others': 6.0
        }
        
        # Dietary habits mapping 
        self.diet_mapping = {
            'Unhealthy': 1,
            'Moderate': 2,
            'Healthy': 3,
            'Others': 2
        }
        
        # Default label encoders for categorical variables
        self.default_encoders = {
            'Gender': {'Male': 0, 'Female': 1, 'Other': 2},
            'Profession': {
                'Student': 0, 'Employee': 1, 'Self-employed': 2, 
                'Unemployed': 3, 'Other': 4
            },
            'Have you ever had suicidal thoughts ?': {'No': 0, 'Yes': 1},
            'Family History of Mental Illness': {'No': 0, 'Yes': 1}
        }
    
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input data for model prediction
        
        Args:
            data: Raw input data dictionary
            
        Returns:
            Processed data dictionary ready for model prediction
        """
        try:
            processed = {}
            
            # Direct numerical features
            numerical_features = [
                'age', 'academic_pressure', 'work_pressure', 'cgpa',
                'study_satisfaction', 'job_satisfaction', 'work_study_hours', 'financial_stress'
            ]
            
            # Feature name mapping to match model expectations
            feature_mapping = {
                'age': 'Age',
                'academic_pressure': 'Academic Pressure',
                'work_pressure': 'Work Pressure',
                'cgpa': 'CGPA',
                'study_satisfaction': 'Study Satisfaction',
                'job_satisfaction': 'Job Satisfaction',
                'work_study_hours': 'Work/Study Hours',
                'financial_stress': 'Financial Stress'
            }
            
            for feature in numerical_features:
                if feature in data:
                    processed[feature_mapping[feature]] = float(data[feature])
                else:
                    logger.warning(f"Missing numerical feature: {feature}")
                    processed[feature_mapping[feature]] = 0.0
            
            # Process sleep duration
            sleep_duration = data.get('sleep_duration', '7-8 hours')
            processed['Sleep_Hours'] = self.sleep_mapping.get(sleep_duration, 7.5)
            
            # Process dietary habits
            dietary_habits = data.get('dietary_habits', 'Moderate')
            processed['Diet_Score'] = self.diet_mapping.get(dietary_habits, 2)
            
            # Encode categorical variables
            gender = data.get('gender', 'Male')
            processed['Gender_encoded'] = self.default_encoders['Gender'].get(gender, 0)
            
            profession = data.get('profession', 'Student')
            processed['Profession_encoded'] = self.default_encoders['Profession'].get(profession, 0)
            
            suicidal_thoughts = data.get('suicidal_thoughts', 'No')
            processed['Have you ever had suicidal thoughts ?_encoded'] = self.default_encoders['Have you ever had suicidal thoughts ?'].get(suicidal_thoughts, 0)
            
            family_history = data.get('family_history', 'No')
            processed['Family History of Mental Illness_encoded'] = self.default_encoders['Family History of Mental Illness'].get(family_history, 0)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(processed)
            processed['Risk_Score'] = risk_score
            
            logger.info("Data preprocessing completed successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise e
    
    def _calculate_risk_score(self, processed_data: Dict[str, Any]) -> float:
        """
        Calculate risk score based on multiple factors
        
        Args:
            processed_data: Processed data dictionary
            
        Returns:
            Calculated risk score
        """
        try:
            risk_score = (
                processed_data.get('Academic Pressure', 0) * 0.2 +
                processed_data.get('Financial Stress', 0) * 0.2 +
                (5 - processed_data.get('Sleep_Hours', 7.5)) * 0.1 +  # Less sleep = higher risk
                (4 - processed_data.get('Diet_Score', 2)) * 0.1 +     # Unhealthy diet = higher risk
                processed_data.get('Have you ever had suicidal thoughts ?_encoded', 0) * 0.3 +
                processed_data.get('Family History of Mental Illness_encoded', 0) * 0.1
            )
            
            return float(risk_score)
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {str(e)}")
            return 0.0
    
    def analyze_risk_factors(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze and identify risk factors from the input data
        
        Args:
            data: Raw input data dictionary
            
        Returns:
            List of risk factors with their impact and description
        """
        risk_factors = []
        
        try:
            # Academic pressure
            academic_pressure = data.get('academic_pressure', 0)
            if academic_pressure >= 4:
                risk_factors.append({
                    'factor': 'High Academic Pressure',
                    'value': academic_pressure,
                    'impact': 'High' if academic_pressure >= 5 else 'Medium',
                    'description': f'Academic pressure level of {academic_pressure}/5 indicates significant stress'
                })
            
            # Financial stress
            financial_stress = data.get('financial_stress', 0)
            if financial_stress >= 4:
                risk_factors.append({
                    'factor': 'High Financial Stress',
                    'value': financial_stress,
                    'impact': 'High' if financial_stress >= 5 else 'Medium',
                    'description': f'Financial stress level of {financial_stress}/5 indicates significant financial pressure'
                })
            
            # Sleep duration
            sleep_duration = data.get('sleep_duration', '7-8 hours')
            if sleep_duration in ['Less than 5 hours', '5-6 hours']:
                risk_factors.append({
                    'factor': 'Insufficient Sleep',
                    'value': sleep_duration,
                    'impact': 'High' if sleep_duration == 'Less than 5 hours' else 'Medium',
                    'description': f'Sleep duration of {sleep_duration} may contribute to mental health issues'
                })
            
            # Dietary habits
            dietary_habits = data.get('dietary_habits', 'Moderate')
            if dietary_habits == 'Unhealthy':
                risk_factors.append({
                    'factor': 'Unhealthy Diet',
                    'value': dietary_habits,
                    'impact': 'Medium',
                    'description': 'Unhealthy dietary habits may negatively impact mental health'
                })
            
            # Suicidal thoughts
            suicidal_thoughts = data.get('suicidal_thoughts', 'No')
            if suicidal_thoughts == 'Yes':
                risk_factors.append({
                    'factor': 'History of Suicidal Thoughts',
                    'value': suicidal_thoughts,
                    'impact': 'Critical',
                    'description': 'Previous suicidal thoughts indicate high risk and require immediate attention'
                })
            
            # Family history
            family_history = data.get('family_history', 'No')
            if family_history == 'Yes':
                risk_factors.append({
                    'factor': 'Family History of Mental Illness',
                    'value': family_history,
                    'impact': 'Medium',
                    'description': 'Family history of mental illness increases risk of developing similar conditions'
                })
            
            # Work pressure
            work_pressure = data.get('work_pressure', 0)
            if work_pressure >= 4:
                risk_factors.append({
                    'factor': 'High Work Pressure',
                    'value': work_pressure,
                    'impact': 'High' if work_pressure >= 5 else 'Medium',
                    'description': f'Work pressure level of {work_pressure}/5 indicates significant workplace stress'
                })
            
            # Low satisfaction levels
            study_satisfaction = data.get('study_satisfaction', 0)
            if study_satisfaction <= 2:
                risk_factors.append({
                    'factor': 'Low Study Satisfaction',
                    'value': study_satisfaction,
                    'impact': 'Medium',
                    'description': f'Study satisfaction level of {study_satisfaction}/5 indicates dissatisfaction with academic life'
                })
            
            job_satisfaction = data.get('job_satisfaction', 0)
            if job_satisfaction <= 2 and job_satisfaction > 0:  # Only if they have a job
                risk_factors.append({
                    'factor': 'Low Job Satisfaction',
                    'value': job_satisfaction,
                    'impact': 'Medium',
                    'description': f'Job satisfaction level of {job_satisfaction}/5 indicates workplace dissatisfaction'
                })
            
            # Low CGPA
            cgpa = data.get('cgpa', 0)
            if cgpa < 6.0:
                risk_factors.append({
                    'factor': 'Low Academic Performance',
                    'value': cgpa,
                    'impact': 'Medium',
                    'description': f'CGPA of {cgpa} may indicate academic struggles affecting mental health'
                })
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Risk factor analysis failed: {str(e)}")
            return []
    
    def generate_recommendations(self, data: Dict[str, Any], probability: float, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """
        Generate personalized recommendations based on the diagnosis and risk factors
        
        Args:
            data: Raw input data dictionary
            probability: Prediction probability
            risk_factors: List of identified risk factors
            
        Returns:
            List of personalized recommendations
        """
        recommendations = []
        
        try:
            # General recommendations based on probability
            if probability > 0.7:
                recommendations.extend([
                    "Seek immediate professional help from a mental health counselor or therapist",
                    "Consider reaching out to a trusted friend or family member for support",
                    "Contact a mental health helpline if you're in crisis"
                ])
            elif probability > 0.4:
                recommendations.extend([
                    "Consider speaking with a mental health professional for assessment",
                    "Focus on stress management techniques and self-care",
                    "Maintain regular sleep and exercise routines"
                ])
            else:
                recommendations.extend([
                    "Continue maintaining good mental health practices",
                    "Stay connected with friends and family",
                    "Engage in activities you enjoy"
                ])
            
            # Specific recommendations based on risk factors
            for risk_factor in risk_factors:
                factor = risk_factor['factor']
                
                if 'Academic Pressure' in factor:
                    recommendations.append("Consider academic counseling or tutoring to manage study stress")
                    recommendations.append("Break down large assignments into smaller, manageable tasks")
                
                elif 'Financial Stress' in factor:
                    recommendations.append("Seek financial counseling or speak with a financial advisor")
                    recommendations.append("Look into scholarships, grants, or part-time work opportunities")
                
                elif 'Sleep' in factor:
                    recommendations.append("Establish a consistent sleep schedule and bedtime routine")
                    recommendations.append("Avoid screens 1 hour before bedtime and create a relaxing environment")
                
                elif 'Diet' in factor:
                    recommendations.append("Focus on a balanced diet with regular meals")
                    recommendations.append("Consider consulting a nutritionist for dietary guidance")
                
                elif 'Suicidal Thoughts' in factor:
                    recommendations.append("URGENT: Contact a mental health professional immediately")
                    recommendations.append("Call a suicide prevention hotline if you're in crisis")
                
                elif 'Family History' in factor:
                    recommendations.append("Be aware of your family history and monitor your mental health regularly")
                    recommendations.append("Consider preventive mental health counseling")
                
                elif 'Work Pressure' in factor:
                    recommendations.append("Discuss workload with your supervisor or academic advisor")
                    recommendations.append("Practice time management and delegation techniques")
                
                elif 'Satisfaction' in factor:
                    recommendations.append("Explore new interests or hobbies to increase life satisfaction")
                    recommendations.append("Consider career counseling or academic guidance")
                
                elif 'Academic Performance' in factor:
                    recommendations.append("Seek academic support services or tutoring")
                    recommendations.append("Meet with academic advisors to discuss study strategies")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec not in seen:
                    seen.add(rec)
                    unique_recommendations.append(rec)
            
            return unique_recommendations[:10]  # Limit to 10 recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return ["Please consult with a mental health professional for personalized advice"]
