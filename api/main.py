#!/usr/bin/env python3
"""
Mental Health Depression Prediction API
FastAPI application for predicting depression risk in students
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import uvicorn
import logging
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.predictor import MentalHealthPredictor
from api.preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Depression Prediction API",
    description="API for predicting depression risk in students using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor and preprocessor
predictor = None
preprocessor = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and preprocessors on startup"""
    global predictor, preprocessor
    try:
        logger.info("Loading models and preprocessors...")
        predictor = MentalHealthPredictor()
        preprocessor = DataPreprocessor()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        logger.error("Please ensure model files are available in the models directory")
        raise e

# Pydantic models for request/response validation
class DiagnosisRequest(BaseModel):
    """Request model for diagnosis endpoint"""
    age: float = Field(..., ge=16, le=100, description="Age of the student (16-100)")
    gender: str = Field(..., description="Gender of the student")
    academic_pressure: float = Field(..., ge=1, le=5, description="Academic pressure level (1-5)")
    work_pressure: float = Field(..., ge=1, le=5, description="Work pressure level (1-5)")
    cgpa: float = Field(..., ge=0, le=10, description="CGPA/GPA (0-10)")
    study_satisfaction: float = Field(..., ge=1, le=5, description="Study satisfaction level (1-5)")
    job_satisfaction: float = Field(..., ge=1, le=5, description="Job satisfaction level (1-5)")
    work_study_hours: float = Field(..., ge=0, le=24, description="Work/Study hours per day (0-24)")
    financial_stress: float = Field(..., ge=1, le=5, description="Financial stress level (1-5)")
    sleep_duration: str = Field(..., description="Sleep duration category")
    dietary_habits: str = Field(..., description="Dietary habits category")
    suicidal_thoughts: str = Field(..., description="History of suicidal thoughts")
    family_history: str = Field(..., description="Family history of mental illness")
    city: str = Field(..., description="City of residence")
    profession: str = Field(..., description="Profession")
    degree: str = Field(..., description="Degree level")
    
    @validator('gender')
    def validate_gender(cls, v):
        valid_genders = ['Male', 'Female', 'Other']
        if v not in valid_genders:
            raise ValueError(f'Gender must be one of: {valid_genders}')
        return v
    
    @validator('sleep_duration')
    def validate_sleep_duration(cls, v):
        valid_sleep = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', 'Others']
        if v not in valid_sleep:
            raise ValueError(f'Sleep duration must be one of: {valid_sleep}')
        return v
    
    @validator('dietary_habits')
    def validate_dietary_habits(cls, v):
        valid_diet = ['Unhealthy', 'Moderate', 'Healthy', 'Others']
        if v not in valid_diet:
            raise ValueError(f'Dietary habits must be one of: {valid_diet}')
        return v
    
    @validator('suicidal_thoughts')
    def validate_suicidal_thoughts(cls, v):
        valid_suicidal = ['Yes', 'No']
        if v not in valid_suicidal:
            raise ValueError(f'Suicidal thoughts must be one of: {valid_suicidal}')
        return v
    
    @validator('family_history')
    def validate_family_history(cls, v):
        valid_family = ['Yes', 'No']
        if v not in valid_family:
            raise ValueError(f'Family history must be one of: {valid_family}')
        return v

class RiskFactor(BaseModel):
    """Model for individual risk factors"""
    factor: str
    value: Any
    impact: str
    description: str

class DiagnosisResponse(BaseModel):
    """Response model for diagnosis endpoint"""
    prediction: int = Field(..., description="Prediction: 0 (Not Depressed) or 1 (Depressed)")
    probability: float = Field(..., ge=0, le=1, description="Probability of depression (0-1)")
    confidence: str = Field(..., description="Confidence level: Low, Medium, or High")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    risk_factors: List[RiskFactor] = Field(..., description="List of identified risk factors")
    recommendations: List[str] = Field(..., description="Recommendations based on the diagnosis")
    timestamp: str = Field(..., description="Timestamp of the prediction")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    model_loaded: bool

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Mental Health Depression Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        model_loaded=predictor is not None
    )

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    """
    Diagnose depression risk based on student data
    
    This endpoint takes student information and returns a depression risk assessment
    including prediction, probability, risk factors, and recommendations.
    """
    try:
        if predictor is None or preprocessor is None:
            raise HTTPException(
                status_code=503, 
                detail="Models not loaded. Please try again later."
            )
        
        logger.info(f"Received diagnosis request for age {request.age}, gender {request.gender}")
        
        # Validate and preprocess the input data
        processed_data = preprocessor.preprocess(request.dict())
        
        # Make prediction
        prediction_result = predictor.predict(processed_data)
        
        # Generate risk factors analysis
        risk_factors = preprocessor.analyze_risk_factors(request.dict())
        
        # Generate recommendations
        recommendations = preprocessor.generate_recommendations(
            request.dict(), 
            prediction_result['probability'],
            risk_factors
        )
        
        # Determine confidence level
        confidence = "High" if prediction_result['probability'] > 0.8 or prediction_result['probability'] < 0.2 else "Medium" if prediction_result['probability'] > 0.6 or prediction_result['probability'] < 0.4 else "Low"
        
        # Determine risk level
        risk_level = "High" if prediction_result['probability'] > 0.7 else "Medium" if prediction_result['probability'] > 0.4 else "Low"
        
        response = DiagnosisResponse(
            prediction=prediction_result['prediction'],
            probability=prediction_result['probability'],
            confidence=confidence,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction completed: {prediction_result['prediction']} (prob: {prediction_result['probability']:.3f})")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return predictor.get_model_info()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
