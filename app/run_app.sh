#!/bin/bash

# Mental Health Diagnoser - Streamlit App Startup Script

echo "🧠 Starting Mental Health Diagnoser Streamlit App..."
echo "=================================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if Streamlit is installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "❌ Streamlit is not installed. Installing dependencies..."
    pip install -r requirements_streamlit.txt
fi

# Check if model files exist
if [ ! -d "models" ]; then
    echo "❌ Models directory not found. Please ensure model files are in the 'models' directory."
    exit 1
fi

# Check for required model files
required_files=(
    "mental_health_model_20250926_165109.pkl"
    "scaler_20250926_165109.pkl"
    "label_encoders_20250926_165109.pkl"
    "feature_columns_20250926_165109.pkl"
)

for file in "${required_files[@]}"; do
    if [ ! -f "models/$file" ]; then
        echo "⚠️  Warning: $file not found in models directory"
    fi
done

echo "✅ Starting Streamlit app..."
echo "🌐 The app will be available at: http://localhost:8501"
echo "📱 Press Ctrl+C to stop the app"
echo ""

# Run the Streamlit app
streamlit run streamlit_app.py --server.headless true --server.port 8501
