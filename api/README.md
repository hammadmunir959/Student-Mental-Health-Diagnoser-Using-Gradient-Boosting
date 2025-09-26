# 🧠 Mental Health Diagnosis API

A FastAPI-based REST API for predicting depression risk in students using machine learning.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python main.py
```

### 3. Access the API
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## 📋 API Endpoints

### POST /diagnose
Predict depression risk based on student data.

**Request Body:**
```json
{
  "age": 20.0,
  "gender": "Female",
  "academic_pressure": 5.0,
  "work_pressure": 4.0,
  "cgpa": 6.2,
  "study_satisfaction": 1.0,
  "job_satisfaction": 1.0,
  "sleep_duration": "Less than 5 hours",
  "dietary_habits": "Unhealthy",
  "work_study_hours": 12.0,
  "financial_stress": 5.0,
  "suicidal_thoughts": "Yes",
  "family_history": "Yes",
  "city": "Delhi",
  "profession": "Student",
  "degree": "Bachelor"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "confidence": "High",
  "risk_level": "High",
  "risk_factors": [
    {
      "factor": "High Academic Pressure",
      "value": 5.0,
      "impact": "High",
      "description": "Academic pressure level of 5.0/5 indicates significant stress"
    }
  ],
  "recommendations": [
    "Seek immediate professional help from a mental health counselor or therapist",
    "Consider reaching out to a trusted friend or family member for support"
  ],
  "timestamp": "2024-01-15T10:30:00"
}
```

### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0",
  "model_loaded": true
}
```

### GET /model-info
Get information about the loaded model.

**Response:**
```json
{
  "model_loaded": true,
  "scaler_loaded": true,
  "encoders_loaded": true,
  "feature_columns": ["Age", "Academic Pressure", ...],
  "num_features": 15,
  "model_name": "Gradient Boosting",
  "accuracy": 0.845,
  "auc": 0.920
}
```

## 📊 Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `age` | float | 16-100 | Age of the student |
| `gender` | string | Male/Female/Other | Gender |
| `academic_pressure` | float | 1-5 | Academic pressure level |
| `work_pressure` | float | 1-5 | Work pressure level |
| `cgpa` | float | 0-10 | CGPA/GPA |
| `study_satisfaction` | float | 1-5 | Study satisfaction level |
| `job_satisfaction` | float | 1-5 | Job satisfaction level |
| `work_study_hours` | float | 0-24 | Work/Study hours per day |
| `financial_stress` | float | 1-5 | Financial stress level |
| `sleep_duration` | string | See options | Sleep duration category |
| `dietary_habits` | string | See options | Dietary habits category |
| `suicidal_thoughts` | string | Yes/No | History of suicidal thoughts |
| `family_history` | string | Yes/No | Family history of mental illness |
| `city` | string | Any | City of residence |
| `profession` | string | Any | Profession |
| `degree` | string | Any | Degree level |

### Valid Options

**Sleep Duration:**
- "Less than 5 hours"
- "5-6 hours"
- "7-8 hours"
- "More than 8 hours"
- "Others"

**Dietary Habits:**
- "Unhealthy"
- "Moderate"
- "Healthy"
- "Others"

## 🧪 Testing

### Test with cURL
```bash
curl -X POST "http://localhost:8000/diagnose" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 20.0,
       "gender": "Female",
       "academic_pressure": 5.0,
       "work_pressure": 4.0,
       "cgpa": 6.2,
       "study_satisfaction": 1.0,
       "job_satisfaction": 1.0,
       "sleep_duration": "Less than 5 hours",
       "dietary_habits": "Unhealthy",
       "work_study_hours": 12.0,
       "financial_stress": 5.0,
       "suicidal_thoughts": "Yes",
       "family_history": "Yes",
       "city": "Delhi",
       "profession": "Student",
       "degree": "Bachelor"
     }'
```

## 🔧 Model Information

- **Model Type**: Gradient Boosting Classifier
- **Accuracy**: 84.5%
- **AUC**: 0.920
- **Features**: 15 engineered features
- **Training Samples**: 22,320
- **Test Samples**: 5,581

## 📁 Project Structure

```
api/
├── main.py                 # FastAPI application
├── predictor.py           # Model loading and prediction
├── preprocessor.py        # Data validation and preprocessing
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🚨 Important Notes

1. **Professional Medical Advice**: This API is for educational and research purposes only. It should not replace professional medical diagnosis or treatment.

2. **Crisis Situations**: If you or someone you know is in immediate danger, please contact emergency services or a crisis helpline.

3. **Data Privacy**: Ensure that any data sent to this API is handled according to privacy regulations and with proper consent.

4. **Model Limitations**: The model is trained on specific data and may not generalize to all populations or situations.

## 🛠️ Development

### Adding New Features
1. Modify the Pydantic models in `main.py` for new input/output fields
2. Update the preprocessor in `preprocessor.py` for new data transformations
3. Update the predictor in `predictor.py` if model loading changes

### Error Handling
The API includes comprehensive error handling for:
- Invalid input data
- Missing model files
- Prediction errors
- Server errors

## 📞 Support

For issues or questions:
1. Check the API documentation at `/docs`
2. Check the logs for error details
