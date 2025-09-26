# Student Mental Health Diagnoser Using Gradient Boosting

A comprehensive machine learning system that predicts depression risk in students using advanced Gradient Boosting algorithms. This project provides both a RESTful API and an interactive Streamlit web application for mental health assessment and diagnosis.

##  Features

- **Advanced ML Model**: Gradient Boosting classifier with 84.5% accuracy
- **Interactive Web App**: Streamlit-based MCQ interface for easy assessment
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Risk Analysis**: Detailed risk factor analysis and recommendations
- **Real-time Predictions**: Instant depression risk assessment
- **Comprehensive Data Processing**: Advanced preprocessing and feature engineering
- **Professional Support**: Integrated Umang Pakistan contact information

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
├── app/                          # Streamlit web application
│   ├── streamlit_app.py          # Main Streamlit application
│   ├── streamlit_utils.py        # Utility functions for the app
│   ├── test_app.py              # Testing script
│   ├── requirements_streamlit.txt # Streamlit dependencies
│   └── run_app.sh               # Startup script
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

# Install Streamlit app dependencies
pip install -r app/requirements_streamlit.txt

# Or install all dependencies
pip install fastapi uvicorn pandas scikit-learn joblib numpy streamlit plotly
```

### 3. Verify Installation

```bash
# Check if all model files are present
ls models/
```

##  Quick Start

### Option 1: Streamlit Web Application (Recommended)

```bash
# Start the Streamlit app
cd app
streamlit run streamlit_app.py

# Or use the startup script
./run_app.sh

# App will be available at http://localhost:8501
```

### Option 2: FastAPI Server

```bash
# Start the FastAPI server
cd api
python main.py

# Server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### Option 3: Test the Application

```bash
# Test the Streamlit app
cd app
python test_app.py
```


## 🌐 Streamlit Web Application

### Features

- **📝 Interactive MCQ Interface**: One question at a time with progress tracking
- **🎯 User-Friendly Design**: Clean, responsive interface that works on all devices
- **📊 Real-time Visualization**: Interactive charts and gauges for risk assessment
- **💡 Personalized Recommendations**: Evidence-based suggestions based on risk factors
- **🔒 Privacy-Focused**: No data stored permanently, all processing in memory
- **📞 Professional Support**: Integrated Umang Pakistan contact information

### Assessment Process

1. **16 Carefully Designed Questions**: Covering all mental health risk factors
2. **Progress Tracking**: Visual progress bar and question navigation
3. **Data Validation**: All inputs validated before proceeding
4. **ML Processing**: Real-time analysis using trained Gradient Boosting model
5. **Comprehensive Results**: Risk level, probability, factors, and recommendations

### Professional Support Integration

The app includes direct contact information for **Umang Pakistan**:
- **Phone**: (92) 0311 7786264
- **Email**: hr@umang.com.pk
- **Website**: www.umang.com.pk

Umang provides online suicide prevention and counselling services with immediate access to clinical psychologists, therapists, and counsellors.

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

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit web application"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app/streamlit_app.py`
   - Click "Deploy"

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0"]
```

### Local Development

```bash
# Run Streamlit app
cd app
streamlit run streamlit_app.py

# Run FastAPI server
cd api
python main.py
```

## 🛠️ Development

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


<<<<<<< HEAD
This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation at `/docs` endpoint

## 🔮 Future Enhancements

- [x] Interactive web application (Streamlit)
- [x] Real-time visualization tools
- [x] Professional support integration
- [ ] Mobile application
- [ ] Integration with health systems
- [ ] Multi-language support
- [ ] Telemedicine integration
- [ ] Real-time monitoring dashboard

## 📚 References

- Mental Health Research Foundation
- WHO Mental Health Guidelines
- Academic studies on student mental health
- Machine Learning best practices

---

**⚠️ Important Disclaimer**: This tool is for educational and research purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for mental health concerns.
