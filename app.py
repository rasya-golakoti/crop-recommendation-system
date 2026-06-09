import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Info box styling */
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .model-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
    
    .feature-bar {
        background-color: #667eea;
        height: 24px;
        border-radius: 12px;
        margin: 8px 0;
        transition: width 0.3s ease;
    }
    
    .feature-label {
        font-weight: 500;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoder with error handling
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("crop_model.pkl", "rb"))
        encoder = pickle.load(open("label_encoder.pkl", "rb"))
        return model, encoder
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'crop_model.pkl' and 'label_encoder.pkl' exist.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.stop()

model, encoder = load_models()

# Model information
MODEL_INFO = {
    "algorithm": "Random Forest Classifier",
    "hyperparameters": {
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "n_estimators": 100,
        "random_state": 42
    },
    "features": ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
    "target_classes": list(encoder.classes_),
    "accuracy": 0.996,
    "cv_score": 0.996
}

# Feature importance data (from your notebook)
feature_importance = {
    'N': 0.09636,
    'P': 0.15085,
    'K': 0.17539,
    'temperature': 0.07238,
    'humidity': 0.22423,
    'ph': 0.05061,
    'rainfall': 0.23018
}

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🌾 Navigation")
    
    # Create navigation buttons
    if st.button("🌱 Crop Prediction", use_container_width=True):
        st.session_state.page = "Crop Prediction"
    if st.button("🤖 Model Details", use_container_width=True):
        st.session_state.page = "Model Details"
    if st.button("📚 Crop Guide", use_container_width=True):
        st.session_state.page = "Crop Guide"
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "About"
    
    # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state.page = "Crop Prediction"
    
    st.markdown("---")
    st.markdown("### 📊 Soil Health Tips")
    with st.expander("ℹ️ Optimal Ranges"):
        st.markdown("""
        - **Nitrogen (N)**: 50-150 mg/kg
        - **Phosphorus (P)**: 30-60 mg/kg
        - **Potassium (K)**: 50-150 mg/kg
        - **Temperature**: 20-35°C
        - **Humidity**: 50-80%
        - **pH**: 6.0-7.5
        - **Rainfall**: 100-200 mm
        """)

# Main content based on selected page
if st.session_state.page == "Crop Prediction":
    st.markdown('<h1 class="main-title">🌾 Smart Crop Recommendation System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box">
        🌱 **Welcome!** Enter your soil and weather parameters below to get AI-powered crop recommendations.
        </div>
        """, unsafe_allow_html=True)
    
    # Create two columns for input layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌿 Soil Nutrients")
        N = st.number_input("🌱 **Nitrogen (N)** (mg/kg)", min_value=0.0, max_value=200.0, value=50.0, step=5.0, help="Ideal range: 50-150 mg/kg")
        P = st.number_input("🪴 **Phosphorus (P)** (mg/kg)", min_value=0.0, max_value=150.0, value=40.0, step=5.0, help="Ideal range: 30-60 mg/kg")
        K = st.number_input("🍂 **Potassium (K)** (mg/kg)", min_value=0.0, max_value=200.0, value=80.0, step=5.0, help="Ideal range: 50-150 mg/kg")
        
        st.markdown("### 🌡️ Weather Conditions")
        temperature = st.number_input("🌡️ **Temperature** (°C)", min_value=0.0, max_value=50.0, value=25.0, step=1.0, help="Ideal range: 20-35°C")
        humidity = st.number_input("💧 **Humidity** (%)", min_value=0.0, max_value=100.0, value=70.0, step=5.0, help="Ideal range: 50-80%")
    
    with col2:
        st.markdown("### 🧪 Soil Chemistry")
        ph = st.number_input("⚗️ **pH Value**", min_value=0.0, max_value=14.0, value=6.5, step=0.1, help="Ideal range: 6.0-7.5")
        rainfall = st.number_input("☔ **Rainfall** (mm/year)", min_value=0.0, max_value=500.0, value=150.0, step=10.0, help="Ideal range: 100-200 mm")
        
        # Input summary with visual indicators
        st.markdown("### 📊 Soil Health Status")
        
        # pH indicator
        if 6.0 <= ph <= 7.5:
            st.success(f"✅ pH: {ph} (Optimal)")
        elif ph < 6.0:
            st.warning(f"⚠️ pH: {ph} (Too acidic - Consider adding lime)")
        else:
            st.warning(f"⚠️ pH: {ph} (Too alkaline - Consider adding sulfur)")
        
        # NPK indicators
        col_n, col_p, col_k = st.columns(3)
        with col_n:
            if 50 <= N <= 150:
                st.success(f"N: {N}")
            else:
                st.warning(f"N: {N}")
        with col_p:
            if 30 <= P <= 60:
                st.success(f"P: {P}")
            else:
                st.warning(f"P: {P}")
        with col_k:
            if 50 <= K <= 150:
                st.success(f"K: {K}")
            else:
                st.warning(f"K: {K}")
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Best Crop", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analyzing soil and weather conditions..."):
            # Create sample dataframe
            sample = pd.DataFrame(
                [[N, P, K, temperature, humidity, ph, rainfall]],
                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            )
            
            # Make prediction
            prediction = model.predict(sample)
            crop = encoder.inverse_transform(prediction)[0]
            
            # Get prediction probabilities if available
            try:
                probabilities = model.predict_proba(sample)[0]
                confidence = max(probabilities) * 100
                confidence_text = f"Confidence: {confidence:.1f}%"
            except:
                confidence_text = ""
            
            # Display result in animated card
            st.markdown(f"""
            <div class="result-card">
                <h2 style="color: white; margin-bottom: 0.5rem;">🌾 Recommended Crop</h2>
                <h1 style="color: white; font-size: 3rem; margin: 0;">{crop.upper()}</h1>
                <p style="color: white; margin-top: 1rem;">✓ Based on your soil and weather conditions</p>
                <p style="color: white; font-size: 0.9rem;">{confidence_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick recommendations
            st.subheader("💡 Recommendations")
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                if ph < 6.0:
                    st.info("📌 Add lime to increase soil pH")
                elif ph > 7.5:
                    st.info("📌 Add sulfur or organic matter to decrease pH")
                else:
                    st.success("✅ Soil pH is in optimal range")
            
            with rec_col2:
                if N < 50:
                    st.info("📌 Apply nitrogen-rich fertilizer (Urea, DAP)")
                if P < 30:
                    st.info("📌 Add phosphorus fertilizer (Single Super Phosphate)")
                if K < 50:
                    st.info("📌 Apply potash fertilizer (Muriate of Potash)")

elif st.session_state.page == "Model Details":
    st.markdown('<h1 class="main-title">🤖 Model Details</h1>', unsafe_allow_html=True)
    
    # Model Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="model-card">
            <h3>📊 Model Performance</h3>
            <p><strong>Algorithm:</strong> {MODEL_INFO['algorithm']}</p>
            <p><strong>Accuracy:</strong> {MODEL_INFO['accuracy']*100:.2f}%</p>
            <p><strong>Cross-validation Score:</strong> {MODEL_INFO['cv_score']*100:.2f}%</p>
            <p><strong>Training Samples:</strong> 1760</p>
            <p><strong>Test Samples:</strong> 440</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="model-card">
            <h3>⚙️ Hyperparameters</h3>
            <p><strong>n_estimators:</strong> {MODEL_INFO['hyperparameters']['n_estimators']}</p>
            <p><strong>max_depth:</strong> {MODEL_INFO['hyperparameters']['max_depth']}</p>
            <p><strong>min_samples_split:</strong> {MODEL_INFO['hyperparameters']['min_samples_split']}</p>
            <p><strong>min_samples_leaf:</strong> {MODEL_INFO['hyperparameters']['min_samples_leaf']}</p>
            <p><strong>random_state:</strong> {MODEL_INFO['hyperparameters']['random_state']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Information with custom progress bars (no pyarrow needed)
    st.markdown("### 📈 Feature Importance")
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Create custom visualization
    for feature, importance in sorted_features:
        st.markdown(f"""
        <div class="feature-label">
            {feature.upper()}: {importance*100:.2f}%
        </div>
        <div style="background-color: #f0f0f0; border-radius: 12px; margin-bottom: 12px;">
            <div class="feature-bar" style="width: {importance*100}%; background: linear-gradient(90deg, #667eea, #764ba2);">
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display importance as a simple table
    st.markdown("### 📋 Feature Importance Table")
    importance_data = []
    for feature, importance in sorted_features:
        importance_data.append([feature.upper(), f"{importance*100:.2f}%"])
    
    # Simple markdown table
    table_md = "| Feature | Importance |\n|---------|------------|\n"
    for row in importance_data:
        table_md += f"| {row[0]} | {row[1]} |\n"
    st.markdown(table_md)
    
    # Crop Classes Information
    st.markdown("### 🌾 Supported Crops")
    st.info(f"This model can recommend from **{len(MODEL_INFO['target_classes'])}** different crops including rice, wheat, maize, cotton, and many more.")
    
    # Grid Search Results
    with st.expander("🔍 Grid Search Cross-Validation Results"):
        st.markdown("""
        The model was optimized using 5-fold cross-validation with the following parameter grid:
        
        - **n_estimators**: 100, 200, 300
        - **max_depth**: None, 10, 20, 30
        - **min_samples_split**: 2, 5, 10
        - **min_samples_leaf**: 1, 2, 4
        
        **Best parameters found:**
        - max_depth: 10
        - min_samples_leaf: 1
        - min_samples_split: 5
        - n_estimators: 100
        """)

elif st.session_state.page == "Crop Guide":
    st.markdown('<h1 class="main-title">📚 Crop Cultivation Guide</h1>', unsafe_allow_html=True)
    
    # Crop database with detailed information
    crop_database = {
        "rice": {
            "temp": "20-37°C",
            "water": "High (150-250mm)",
            "soil": "Clay loam",
            "season": "Kharif (June-Nov)",
            "fertilizer": "N:P:K - 120:60:40",
            "duration": "120-150 days"
        },
        "wheat": {
            "temp": "10-25°C",
            "water": "Moderate (100-150mm)",
            "soil": "Well-drained loamy",
            "season": "Rabi (Oct-Mar)",
            "fertilizer": "N:P:K - 120:60:40",
            "duration": "110-130 days"
        },
        "maize": {
            "temp": "21-27°C",
            "water": "Moderate (120-180mm)",
            "soil": "Well-drained fertile",
            "season": "Kharif (June-Oct)",
            "fertilizer": "N:P:K - 150:60:40",
            "duration": "80-110 days"
        },
        "cotton": {
            "temp": "21-30°C",
            "water": "Low (50-100mm)",
            "soil": "Black cotton soil",
            "season": "Kharif (June-Oct)",
            "fertilizer": "N:P:K - 100:50:50",
            "duration": "150-180 days"
        }
    }
    
    # Get unique crop names from encoder
    crop_list = list(encoder.classes_)
    
    selected_crop = st.selectbox("Select a crop to learn more:", crop_list)
    
    if selected_crop:
        if selected_crop in crop_database:
            info = crop_database[selected_crop]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🌡️ Temperature", info["temp"])
                st.metric("🌱 Soil Type", info["soil"])
            with col2:
                st.metric("💧 Water Need", info["water"])
                st.metric("🧪 Fertilizer", info["fertilizer"])
            with col3:
                st.metric("📅 Growing Season", info["season"])
                st.metric("📆 Harvest Duration", info["duration"])
        else:
            st.markdown(f"""
            <div class="info-box">
            <h3>🌱 {selected_crop.upper()}</h3>
            <p>For detailed cultivation practices for {selected_crop}, please consult local agricultural experts or agricultural extension services.</p>
            </div>
            """, unsafe_allow_html=True)

else:  # About section
    st.markdown('<h1 class="main-title">ℹ️ About This System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🌾")
    
    with col2:
        st.markdown("""
        ### 🌾 Smart Crop Recommendation System
        
        This AI-powered system helps farmers make data-driven decisions about crop selection based on:
        
        - **Soil Parameters**: Nitrogen, Phosphorus, Potassium levels, and pH
        - **Weather Conditions**: Temperature, Humidity, and Rainfall
        
        ### 🎯 Benefits
        - Maximize crop yield
        - Reduce input costs
        - Promote sustainable farming
        - Minimize crop failure risk
        
        ### 🤖 Technology Stack
        - **Model**: Random Forest Classifier (optimized via GridSearchCV)
        - **Accuracy**: 99.6% on test data
        - **Training Data**: 2200 samples with 7 features
        - **Framework**: scikit-learn, pandas, numpy
        
        ### 📞 Support
        For issues or suggestions, please contact our support team.
        """)
    
    st.markdown("---")
    st.markdown("### 📊 System Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Model Accuracy", "99.6%", delta="High")
    with col2:
        st.metric("🌾 Supported Crops", str(len(encoder.classes_)), delta=f"{len(encoder.classes_)} Crops")
    with col3:
        st.metric("⚡ Response Time", "<1 sec", delta="Fast")