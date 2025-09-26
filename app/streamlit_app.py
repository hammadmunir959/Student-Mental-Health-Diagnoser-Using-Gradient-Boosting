#!/usr/bin/env python3
"""
Mental Health Diagnoser - Streamlit App
Interactive MCQ-based mental health assessment tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.predictor import MentalHealthPredictor
from api.preprocessor import DataPreprocessor

# Page configuration
st.set_page_config(
    page_title="Mental Health Diagnoser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .result-container {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .recommendation {
        background-color: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #28a745;
        color: #2c3e50;
        font-weight: 500;
    }
    .risk-factor {
        background-color: #fff3cd;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
        color: #856404;
        font-weight: 500;
    }
    .risk-factor-title {
        color: #d63384;
        font-weight: bold;
        font-size: 1.1em;
    }
    .recommendation-title {
        color: #198754;
        font-weight: bold;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'prediction_complete' not in st.session_state:
    st.session_state.prediction_complete = False

# Load models
@st.cache_resource
def load_models():
    """Load the ML models and preprocessor"""
    try:
        predictor = MentalHealthPredictor()
        preprocessor = DataPreprocessor()
        return predictor, preprocessor
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None

# Define questions
QUESTIONS = [
    {
        "id": "age",
        "question": "What is your age?",
        "type": "number_input",
        "min_value": 16,
        "max_value": 100,
        "value": 20,
        "help": "Please enter your current age"
    },
    {
        "id": "gender",
        "question": "What is your gender?",
        "type": "selectbox",
        "options": ["Male", "Female", "Other"],
        "help": "Please select your gender identity"
    },
    {
        "id": "academic_pressure",
        "question": "How would you rate your academic pressure level?",
        "type": "slider",
        "min_value": 1,
        "max_value": 5,
        "value": 3,
        "help": "1 = Very Low, 5 = Very High",
        "labels": {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
    },
    {
        "id": "work_pressure",
        "question": "How would you rate your work pressure level?",
        "type": "slider",
        "min_value": 1,
        "max_value": 5,
        "value": 3,
        "help": "1 = Very Low, 5 = Very High",
        "labels": {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
    },
    {
        "id": "cgpa",
        "question": "What is your current CGPA/GPA?",
        "type": "number_input",
        "min_value": 0.0,
        "max_value": 10.0,
        "value": 7.0,
        "step": 0.1,
        "help": "Please enter your current Grade Point Average (0-10 scale)"
    },
    {
        "id": "study_satisfaction",
        "question": "How satisfied are you with your studies?",
        "type": "slider",
        "min_value": 1,
        "max_value": 5,
        "value": 3,
        "help": "1 = Very Dissatisfied, 5 = Very Satisfied",
        "labels": {1: "Very Dissatisfied", 2: "Dissatisfied", 3: "Neutral", 4: "Satisfied", 5: "Very Satisfied"}
    },
    {
        "id": "job_satisfaction",
        "question": "How satisfied are you with your job? (If not employed, select 'Not Applicable')",
        "type": "selectbox",
        "options": ["Not Applicable", "1 - Very Dissatisfied", "2 - Dissatisfied", "3 - Neutral", "4 - Satisfied", "5 - Very Satisfied"],
        "help": "Select your job satisfaction level or 'Not Applicable' if you don't have a job"
    },
    {
        "id": "work_study_hours",
        "question": "How many hours per day do you spend on work/study?",
        "type": "slider",
        "min_value": 0,
        "max_value": 24,
        "value": 8,
        "help": "Total hours spent on work and study activities per day"
    },
    {
        "id": "financial_stress",
        "question": "How would you rate your financial stress level?",
        "type": "slider",
        "min_value": 1,
        "max_value": 5,
        "value": 3,
        "help": "1 = Very Low, 5 = Very High",
        "labels": {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
    },
    {
        "id": "sleep_duration",
        "question": "How many hours do you typically sleep per night?",
        "type": "selectbox",
        "options": ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"],
        "help": "Select your typical sleep duration"
    },
    {
        "id": "dietary_habits",
        "question": "How would you describe your dietary habits?",
        "type": "selectbox",
        "options": ["Unhealthy", "Moderate", "Healthy", "Others"],
        "help": "Please select the option that best describes your eating habits"
    },
    {
        "id": "suicidal_thoughts",
        "question": "Have you ever had suicidal thoughts?",
        "type": "selectbox",
        "options": ["No", "Yes"],
        "help": "This information is confidential and will be used only for assessment purposes"
    },
    {
        "id": "family_history",
        "question": "Do you have a family history of mental illness?",
        "type": "selectbox",
        "options": ["No", "Yes"],
        "help": "Please select if any family members have been diagnosed with mental health conditions"
    },
    {
        "id": "city",
        "question": "Which city do you currently live in?",
        "type": "text_input",
        "help": "Enter your current city of residence"
    },
    {
        "id": "profession",
        "question": "What is your current profession?",
        "type": "selectbox",
        "options": ["Student", "Employee", "Self-employed", "Unemployed", "Other"],
        "help": "Please select your current professional status"
    },
    {
        "id": "degree",
        "question": "What is your highest educational degree?",
        "type": "selectbox",
        "options": ["High School", "Bachelor's", "Master's", "PhD", "Other"],
        "help": "Please select your highest completed educational level"
    }
]

def render_question(question_data):
    """Render a single question based on its type"""
    question_id = question_data["id"]
    question_text = question_data["question"]
    question_type = question_data["type"]
    help_text = question_data.get("help", "")
    
    st.markdown(f'<div class="question-container">', unsafe_allow_html=True)
    st.markdown(f"**Question {st.session_state.current_question + 1} of {len(QUESTIONS)}**")
    st.markdown(f"### {question_text}")
    
    if help_text:
        st.info(help_text)
    
    # Render input based on type
    if question_type == "number_input":
        value = st.number_input(
            "Your answer:",
            min_value=question_data.get("min_value", 0),
            max_value=question_data.get("max_value", 100),
            value=question_data.get("value", 0),
            step=question_data.get("step", 1),
            key=f"input_{question_id}"
        )
        st.session_state.answers[question_id] = value
    
    elif question_type == "slider":
        labels = question_data.get("labels", None)
        value = st.slider(
            "Your answer:",
            min_value=question_data.get("min_value", 1),
            max_value=question_data.get("max_value", 5),
            value=question_data.get("value", 3),
            key=f"input_{question_id}",
            format="%d" if labels is None else None
        )
        if labels:
            st.write(f"**Selected:** {labels[value]}")
        st.session_state.answers[question_id] = value
    
    elif question_type == "selectbox":
        options = question_data["options"]
        value = st.selectbox(
            "Your answer:",
            options=options,
            key=f"input_{question_id}"
        )
        st.session_state.answers[question_id] = value
    
    elif question_type == "text_input":
        value = st.text_input(
            "Your answer:",
            key=f"input_{question_id}",
            placeholder="Enter your answer here..."
        )
        st.session_state.answers[question_id] = value
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_job_satisfaction(job_satisfaction_str):
    """Process job satisfaction string to numeric value"""
    if job_satisfaction_str == "Not Applicable":
        return 0
    else:
        # Extract number from string like "1 - Very Dissatisfied"
        return int(job_satisfaction_str.split(" - ")[0])

def make_prediction():
    """Make prediction using the loaded model"""
    try:
        # Load models
        predictor, preprocessor = load_models()
        if predictor is None or preprocessor is None:
            st.error("Models could not be loaded. Please check the model files.")
            return None
        
        # Process answers
        answers = st.session_state.answers.copy()
        
        # Preprocess data
        processed_data = preprocessor.preprocess(answers)
        
        # Make prediction
        prediction_result = predictor.predict(processed_data)
        
        # Analyze risk factors
        risk_factors = preprocessor.analyze_risk_factors(answers)
        
        # Generate recommendations
        recommendations = preprocessor.generate_recommendations(
            answers, 
            prediction_result['probability'],
            risk_factors
        )
        
        # Determine confidence and risk levels
        probability = prediction_result['probability']
        confidence = "High" if probability > 0.8 or probability < 0.2 else "Medium" if probability > 0.6 or probability < 0.4 else "Low"
        risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        
        return {
            'prediction': prediction_result['prediction'],
            'probability': probability,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

def display_results(result):
    """Display prediction results"""
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("## 🧠 Mental Health Assessment Results")
    st.markdown(f"**Assessment completed on:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    
    # Prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_text = "Depression Risk Detected" if result['prediction'] == 1 else "No Depression Risk"
        prediction_color = "🔴" if result['prediction'] == 1 else "🟢"
        st.metric("Assessment Result", f"{prediction_color} {prediction_text}")
    
    with col2:
        probability_percent = result['probability'] * 100
        st.metric("Risk Probability", f"{probability_percent:.1f}%")
    
    with col3:
        risk_class = f"risk-{result['risk_level'].lower()}"
        st.markdown(f"**Risk Level:** <span class='{risk_class}'>{result['risk_level']}</span>", unsafe_allow_html=True)
    
    # Risk level explanation
    st.markdown("### Risk Level Explanation")
    if result['risk_level'] == 'High':
        st.warning("⚠️ **High Risk**: The assessment indicates a high probability of depression risk. Please consider seeking professional help.")
    elif result['risk_level'] == 'Medium':
        st.info("ℹ️ **Medium Risk**: The assessment indicates a moderate probability of depression risk. Consider monitoring your mental health and seeking support if needed.")
    else:
        st.success("✅ **Low Risk**: The assessment indicates a low probability of depression risk. Continue maintaining good mental health practices.")
    
    # Risk factors
    if result['risk_factors']:
        st.markdown("### 🔍 Identified Risk Factors")
        for i, factor in enumerate(result['risk_factors'], 1):
            impact_color = {
                'Critical': '🔴',
                'High': '🟠', 
                'Medium': '🟡',
                'Low': '🟢'
            }.get(factor['impact'], '⚪')
            
            st.markdown(f'<div class="risk-factor">', unsafe_allow_html=True)
            st.markdown(f'<div class="risk-factor-title">{i}. {factor["factor"]} {impact_color}</div>', unsafe_allow_html=True)
            st.markdown(f"**Impact:** {factor['impact']}")
            st.markdown(f"**Description:** {factor['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 💡 Personalized Recommendations")
    for i, recommendation in enumerate(result['recommendations'], 1):
        st.markdown(f'<div class="recommendation"><div class="recommendation-title">{i}.</div> {recommendation}</div>', unsafe_allow_html=True)
    
    # Visualization
    st.markdown("### 📊 Risk Assessment Visualization")
    
    # Create risk level gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = result['probability'] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Depression Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
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
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("### ⚠️ Important Disclaimer")
    st.warning("""
    **This assessment is for educational and research purposes only.** 
    It should not replace professional medical advice, diagnosis, or treatment. 
    If you're experiencing mental health concerns, please consult with qualified healthcare professionals.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Diagnoser</h1>', unsafe_allow_html=True)
    st.markdown("### A comprehensive mental health assessment tool using machine learning")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Assessment Progress")
        progress = (st.session_state.current_question + 1) / len(QUESTIONS)
        st.progress(progress)
        st.markdown(f"**Progress:** {st.session_state.current_question + 1} / {len(QUESTIONS)} questions")
        
        st.markdown("---")
        st.markdown("## ℹ️ About This Tool")
        st.markdown("""
        This tool uses advanced machine learning to assess mental health risk factors based on your responses to carefully designed questions.
        
        **Features:**
        - ✅ Evidence-based assessment
        - 🔒 Privacy-focused (no data stored)
        - 📊 Detailed risk analysis
        - 💡 Personalized recommendations
        """)
        
        st.markdown("---")
        st.markdown("## 🆘 Need Help?")
        st.markdown("""
        **Contact Umang Pakistan**
        
        Umang is strictly an online suicide preventive and counselling service providing immediate access to clinical psychologists/therapists/counsellors in most cases.
        
        **📞 PHONE:** (92) 0311 7786264
        
        **📧 EMAIL:** hr@umang.com.pk
        
        **🌐 WEBSITE:** www.umang.com.pk
        """)
    
    # Main content
    if not st.session_state.prediction_complete:
        # Show current question
        if st.session_state.current_question < len(QUESTIONS):
            current_question_data = QUESTIONS[st.session_state.current_question]
            render_question(current_question_data)
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("⬅️ Previous", disabled=st.session_state.current_question == 0):
                    st.session_state.current_question -= 1
                    st.rerun()
            
            with col2:
                if st.button("Reset 🔄"):
                    st.session_state.current_question = 0
                    st.session_state.answers = {}
                    st.session_state.prediction_complete = False
                    st.rerun()
            
            with col3:
                if st.button("Next ➡️", disabled=st.session_state.current_question == len(QUESTIONS) - 1):
                    # Validate current answer
                    current_id = current_question_data["id"]
                    if current_id in st.session_state.answers:
                        st.session_state.current_question += 1
                        st.rerun()
                    else:
                        st.warning("Please answer the current question before proceeding.")
                elif st.button("Complete Assessment 🎯", disabled=st.session_state.current_question != len(QUESTIONS) - 1):
                    # Validate all answers
                    missing_answers = []
                    for q in QUESTIONS:
                        if q["id"] not in st.session_state.answers:
                            missing_answers.append(q["id"])
                    
                    if missing_answers:
                        st.error(f"Please answer all questions. Missing: {', '.join(missing_answers)}")
                    else:
                        # Make prediction
                        with st.spinner("Analyzing your responses..."):
                            result = make_prediction()
                            if result:
                                st.session_state.prediction_complete = True
                                st.session_state.prediction_result = result
                                st.rerun()
        
        # Show summary of answers
        if st.session_state.answers:
            with st.expander("📝 Review Your Answers"):
                for i, question in enumerate(QUESTIONS):
                    if question["id"] in st.session_state.answers:
                        answer = st.session_state.answers[question["id"]]
                        st.write(f"**{i+1}. {question['question']}**")
                        st.write(f"   Answer: {answer}")
                        st.write("---")
    
    else:
        # Show results
        if 'prediction_result' in st.session_state:
            display_results(st.session_state.prediction_result)
            
            # Action buttons
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔄 Take Assessment Again"):
                    st.session_state.current_question = 0
                    st.session_state.answers = {}
                    st.session_state.prediction_complete = False
                    if 'prediction_result' in st.session_state:
                        del st.session_state.prediction_result
                    st.rerun()
            
            with col2:
                if st.button("ℹ️ About This Tool"):
                    st.info("This tool uses machine learning to assess mental health risk factors. Results are for educational purposes only and should not replace professional medical advice.")

if __name__ == "__main__":
    main()
