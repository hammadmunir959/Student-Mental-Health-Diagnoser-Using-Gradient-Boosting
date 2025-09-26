#!/usr/bin/env python3
"""
Streamlit Utilities for Mental Health Diagnoser
Helper functions for the Streamlit app
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def validate_answers(answers: Dict[str, Any], questions: List[Dict]) -> List[str]:
    """
    Validate that all required questions have been answered
    
    Args:
        answers: Dictionary of user answers
        questions: List of question dictionaries
        
    Returns:
        List of missing question IDs
    """
    missing = []
    for question in questions:
        if question["id"] not in answers:
            missing.append(question["id"])
        elif question["type"] == "text_input" and not answers[question["id"]]:
            missing.append(question["id"])
    
    return missing

def format_risk_level(risk_level: str) -> str:
    """
    Format risk level with appropriate styling
    
    Args:
        risk_level: Risk level string (Low, Medium, High)
        
    Returns:
        Formatted risk level string
    """
    risk_icons = {
        "Low": "🟢",
        "Medium": "🟡", 
        "High": "🔴"
    }
    
    return f"{risk_icons.get(risk_level, '⚪')} {risk_level} Risk"

def format_confidence(confidence: str) -> str:
    """
    Format confidence level with appropriate styling
    
    Args:
        confidence: Confidence level string (Low, Medium, High)
        
    Returns:
        Formatted confidence level string
    """
    confidence_icons = {
        "Low": "🔸",
        "Medium": "🔹",
        "High": "🔷"
    }
    
    return f"{confidence_icons.get(confidence, '⚪')} {confidence} Confidence"

def create_risk_factors_chart(risk_factors: List[Dict[str, Any]]) -> Optional[object]:
    """
    Create a chart showing risk factors and their impact
    
    Args:
        risk_factors: List of risk factor dictionaries
        
    Returns:
        Plotly chart object or None
    """
    if not risk_factors:
        return None
    
    try:
        import plotly.express as px
        
        # Prepare data for chart
        factors_data = []
        for factor in risk_factors:
            impact_scores = {
                'Critical': 4,
                'High': 3,
                'Medium': 2,
                'Low': 1
            }
            
            factors_data.append({
                'Factor': factor['factor'],
                'Impact': factor['impact'],
                'Impact_Score': impact_scores.get(factor['impact'], 1),
                'Description': factor['description'][:50] + "..." if len(factor['description']) > 50 else factor['description']
            })
        
        # Create horizontal bar chart
        df = pd.DataFrame(factors_data)
        fig = px.bar(
            df, 
            x='Impact_Score', 
            y='Factor',
            color='Impact',
            color_discrete_map={
                'Critical': '#d62728',
                'High': '#ff7f0e', 
                'Medium': '#ffdd57',
                'Low': '#2ca02c'
            },
            title="Risk Factors Impact Analysis",
            labels={'Impact_Score': 'Impact Level', 'Factor': 'Risk Factor'},
            hover_data=['Description']
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            xaxis_title="Impact Level",
            yaxis_title="Risk Factors"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create risk factors chart: {str(e)}")
        return None

def create_probability_gauge(probability: float) -> object:
    """
    Create a gauge chart showing depression risk probability
    
    Args:
        probability: Probability value between 0 and 1
        
    Returns:
        Plotly gauge chart object
    """
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Depression Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def get_recommendation_priority(recommendation: str) -> int:
    """
    Get priority level for recommendations based on keywords
    
    Args:
        recommendation: Recommendation text
        
    Returns:
        Priority level (1=High, 2=Medium, 3=Low)
    """
    high_priority_keywords = [
        'urgent', 'immediate', 'crisis', 'professional help', 
        'therapist', 'counselor', 'helpline'
    ]
    
    medium_priority_keywords = [
        'consider', 'recommend', 'suggest', 'important', 
        'monitor', 'regular', 'routine'
    ]
    
    recommendation_lower = recommendation.lower()
    
    if any(keyword in recommendation_lower for keyword in high_priority_keywords):
        return 1
    elif any(keyword in recommendation_lower for keyword in medium_priority_keywords):
        return 2
    else:
        return 3

def sort_recommendations_by_priority(recommendations: List[str]) -> List[str]:
    """
    Sort recommendations by priority level
    
    Args:
        recommendations: List of recommendation strings
        
    Returns:
        Sorted list of recommendations
    """
    return sorted(recommendations, key=get_recommendation_priority)

def create_progress_summary(answers: Dict[str, Any], questions: List[Dict]) -> Dict[str, Any]:
    """
    Create a summary of user progress through the assessment
    
    Args:
        answers: Dictionary of user answers
        questions: List of question dictionaries
        
    Returns:
        Summary dictionary with progress statistics
    """
    total_questions = len(questions)
    answered_questions = len(answers)
    progress_percentage = (answered_questions / total_questions) * 100
    
    # Categorize questions by type
    question_types = {}
    for question in questions:
        q_type = question['type']
        if q_type not in question_types:
            question_types[q_type] = 0
        question_types[q_type] += 1
    
    return {
        'total_questions': total_questions,
        'answered_questions': answered_questions,
        'progress_percentage': progress_percentage,
        'question_types': question_types,
        'is_complete': answered_questions == total_questions
    }

def export_results_to_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export prediction results to a dictionary format for potential saving
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        Exportable dictionary
    """
    export_data = {
        'timestamp': result.get('timestamp', ''),
        'prediction': result.get('prediction', 0),
        'probability': result.get('probability', 0.0),
        'confidence': result.get('confidence', 'Unknown'),
        'risk_level': result.get('risk_level', 'Unknown'),
        'risk_factors': result.get('risk_factors', []),
        'recommendations': result.get('recommendations', []),
        'assessment_version': '1.0.0'
    }
    
    return export_data
