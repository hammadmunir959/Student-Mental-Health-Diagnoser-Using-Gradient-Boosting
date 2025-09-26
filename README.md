# Student Mental Health Diagnoser Using Gradient Boosting

A comprehensive machine learning system that predicts depression risk in students using advanced Gradient Boosting algorithms. This project provides a RESTful API for mental health assessment and diagnosis.

##  Features

- **Advanced ML Model**: Gradient Boosting classifier with 84.5% accuracy
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Risk Analysis**: Detailed risk factor analysis and recommendations
- **Real-time Predictions**: Instant depression risk assessment
- **Comprehensive Data Processing**: Advanced preprocessing and feature engineering

##  Model Performance

- **Accuracy**: 84.54%
- **AUC Score**: 0.9201
- **Precision**: 85.80%
- **Recall**: 88.19%
- **F1-Score**: 86.98%
- **Training Samples**: 22,320
- **Test Samples**: 5,581

##  Project Structure

```
mental_health_diagnoser/
├── api/                           # FastAPI backend
│   ├── main.py                   # Main API server with endpoints
│   ├── predictor.py              # ML model prediction logic
│   ├── preprocessor.py           # Data preprocessing and feature engineering
│   └── requirements.txt          # API dependencies
├── data/                         # Datasets and processed data
│   ├── eda_complete.csv          # Complete EDA dataset
│   ├── preprocessed_data.csv     # Preprocessed training data
│   ├── feature_columns.csv       # Feature column definitions
│   └── Student Depression Dataset.csv  # Original dataset
├── models/                       # Trained ML models and artifacts
│   ├── mental_health_model_20250926_165109.pkl      # Main Gradient Boosting model
│   ├── scaler_20250926_165109.pkl                   # Feature scaler
│   ├── label_encoders_20250926_165109.pkl           # Categorical encoders
│   ├── feature_columns_20250926_165109.pkl          # Feature column mappings
│   └── model_metadata_20250926_165109.json          # Model performance metrics
├── notebooks/                    # Jupyter notebooks for analysis
│   └── diagnoser.ipynb          # Complete EDA and model development
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

##  Installation & Setup

### Prerequisites

- Python 3.8+
- pip or conda
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mental_health_diagnoser
```

### 2. Install Dependencies

```bash
# Install API dependencies
pip install -r api/requirements.txt

# Or install all dependencies
pip install fastapi uvicorn pandas scikit-learn joblib numpy streamlit plotly
```

### 3. Verify Installation

```bash
# Check if all model files are present
ls models/
```

##  Quick Start

### Start the API Server

```bash
# Start the FastAPI server
cd api
python main.py

# Server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```


##  API Endpoints

### Core Endpoints

- `GET /` - API information and status
- `GET /health` - Health check endpoint
- `POST /diagnose` - Main diagnosis endpoint
- `GET /model-info` - Model information and metrics

### Usage Example

```python
import requests

# Example API call
url = "http://localhost:8000/diagnose"
data = {
    "age": 22,
    "gender": "Female",
    "academic_pressure": 4,
    "work_pressure": 3,
    "cgpa": 8.5,
    "study_satisfaction": 3,
    "job_satisfaction": 4,
    "work_study_hours": 6,
    "financial_stress": 2,
    "sleep_duration": "7-8 hours",
    "dietary_habits": "Healthy",
    "suicidal_thoughts": "No",
    "family_history": "No",
    "city": "Mumbai",
    "profession": "Student",
    "degree": "Bachelor's"
}

response = requests.post(url, json=data)
result = response.json()
```

##  Input Features

The model analyzes the following student characteristics:

### Personal Information
- **Age**: Student's age (16-100)
- **Gender**: Male, Female, Other
- **City**: City of residence
- **Profession**: Current profession
- **Degree**: Educational level

### Academic & Work Factors
- **Academic Pressure**: Level of academic stress (1-5)
- **Work Pressure**: Level of work-related stress (1-5)
- **CGPA**: Current Grade Point Average (0-10)
- **Study Satisfaction**: Satisfaction with studies (1-5)
- **Job Satisfaction**: Satisfaction with job (1-5)
- **Work/Study Hours**: Daily hours spent on work/study (0-24)

### Lifestyle & Health
- **Financial Stress**: Level of financial pressure (1-5)
- **Sleep Duration**: Sleep pattern category
- **Dietary Habits**: Diet quality assessment
- **Suicidal Thoughts**: History of suicidal ideation
- **Family History**: Family history of mental illness

##  Model Details

### Algorithm: Gradient Boosting
- **Base Estimators**: Decision Trees
- **Learning Rate**: Optimized through cross-validation
- **Max Depth**: Tuned for optimal performance
- **Feature Engineering**: Advanced preprocessing pipeline

### Feature Engineering
- **Numerical Scaling**: StandardScaler normalization
- **Categorical Encoding**: Label encoding for categorical variables
- **Feature Selection**: Optimized feature set of 15 key indicators
- **Risk Score Calculation**: Composite risk assessment

### Performance Metrics
- **Cross-Validation**: 5-fold stratified CV
- **Train-Test Split**: 80-20 split
- **Class Balance**: Handled through stratified sampling
- **Feature Importance**: Analyzed for model interpretability

##  Risk Assessment

The system provides comprehensive risk analysis:

### Risk Levels
- **Low Risk**: Probability < 0.4
- **Medium Risk**: Probability 0.4 - 0.7
- **High Risk**: Probability > 0.7

### Risk Factors Analysis
- Identifies key contributing factors
- Provides impact assessment for each factor
- Offers detailed explanations for risk factors

### Recommendations
- Personalized recommendations based on risk profile
- Evidence-based suggestions for mental health improvement
- Resource links and professional guidance

##  Development

### Model Training

The model was trained using a comprehensive dataset with the following process:

1. **Data Collection**: Student mental health survey data
2. **Exploratory Data Analysis**: Comprehensive EDA in Jupyter notebook
3. **Feature Engineering**: Advanced preprocessing pipeline
4. **Model Selection**: Comparison of multiple algorithms
5. **Hyperparameter Tuning**: Grid search optimization
6. **Validation**: Cross-validation and holdout testing

### Adding New Features

To add new features to the model:

1. Update the preprocessing pipeline in `api/preprocessor.py`
2. Retrain the model with new features
3. Update the API validation schemas
4. Test the new features thoroughly

##  Data Privacy & Ethics

- **Anonymized Data**: All personal identifiers removed
- **Consent-Based**: Data collected with proper consent
- **Secure Processing**: No data stored permanently
- **Ethical Guidelines**: Follows mental health research ethics

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request



---

**⚠️ Important Disclaimer**: This tool is for educational and research purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for mental health concerns.
