import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Page setup
st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="wide")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Title
st.title("🌾 Smart Crop Recommendation System")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    model = pickle.load(open("crop_model.pkl", "rb"))
    encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, encoder

model, encoder = load_models()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 Soil Nutrients")
    N = st.number_input("Nitrogen (N) - mg/kg", min_value=0.0, max_value=200.0, value=50.0, help="Ideal range: 50-150")
    P = st.number_input("Phosphorus (P) - mg/kg", min_value=0.0, max_value=150.0, value=40.0, help="Ideal range: 30-60")
    K = st.number_input("Potassium (K) - mg/kg", min_value=0.0, max_value=200.0, value=80.0, help="Ideal range: 50-150")
    
    st.subheader("🌡️ Weather Conditions")
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, help="Ideal range: 20-35")
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, help="Ideal range: 50-80")

with col2:
    st.subheader("🧪 Soil Chemistry")
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5, help="Ideal range: 6.0-7.5")
    rainfall = st.number_input("Rainfall (mm/year)", min_value=0.0, max_value=500.0, value=150.0, help="Ideal range: 100-200")
    
    # Enhanced input summary with visual indicators
    st.subheader("📊 Soil Health Status")
    
    # pH indicator
    if 6.0 <= ph <= 7.5:
        st.success(f"✅ pH: {ph} (Optimal)")
    elif ph < 6.0:
        st.warning(f"⚠️ pH: {ph} (Too acidic - Add lime)")
    else:
        st.warning(f"⚠️ pH: {ph} (Too alkaline - Add sulfur)")
    
    # NPK indicators
    col_n, col_p, col_k = st.columns(3)
    with col_n:
        if N >= 50:
            st.success(f"N: {N}")
        else:
            st.warning(f"N: {N}")
    with col_p:
        if P >= 30:
            st.success(f"P: {P}")
        else:
            st.warning(f"P: {P}")
    with col_k:
        if K >= 50:
            st.success(f"K: {K}")
        else:
            st.warning(f"K: {K}")

# Prediction button
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict = st.button("🔮 Predict Crop", use_container_width=True)

# Make prediction
if predict:
    with st.spinner("Analyzing soil and weather conditions..."):
        # Create input dataframe
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Predict
        prediction = model.predict(input_data)
        crop = encoder.inverse_transform(prediction)[0]
        
        # Save to history
        st.session_state.history.append({
            "Crop": crop,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "N": N, "P": P, "K": K,
            "Temp": temperature, "Humidity": humidity,
            "pH": ph, "Rainfall": rainfall
        })
        
        # Show result
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 1rem;
            text-align: center;
            margin-top: 2rem;
        ">
            <h2 style="color: white;">🌾 Recommended Crop</h2>
            <h1 style="color: white; font-size: 3rem;">{crop.upper()}</h1>
            <p style="color: white;">✓ Based on your soil and weather conditions</p>
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

# Show history in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📜 Recent Predictions")
    if st.session_state.history:
        for entry in st.session_state.history[-5:]:
            st.write(f"• {entry['Crop'].upper()} ({entry['Date']})")
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No predictions yet")
    
    st.markdown("---")
    st.caption("💡 Tip: Optimal ranges are shown as tooltips on input fields")