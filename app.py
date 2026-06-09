import streamlit as st
import pandas as pd
import pickle

# Page setup
st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="wide")

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

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 Soil Nutrients")
    N = st.number_input("Nitrogen (N) - mg/kg", min_value=0.0, max_value=200.0, value=50.0)
    P = st.number_input("Phosphorus (P) - mg/kg", min_value=0.0, max_value=150.0, value=40.0)
    K = st.number_input("Potassium (K) - mg/kg", min_value=0.0, max_value=200.0, value=80.0)
    
    st.subheader("🌡️ Weather Conditions")
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)

with col2:
    st.subheader("🧪 Soil Chemistry")
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm/year)", min_value=0.0, max_value=500.0, value=150.0)
    
    # Show input summary
    st.subheader("📊 Input Summary")
    st.write(f"**N:** {N} | **P:** {P} | **K:** {K}")
    st.write(f"**Temp:** {temperature}°C | **Humidity:** {humidity}%")
    st.write(f"**pH:** {ph} | **Rainfall:** {rainfall} mm")

# Prediction button
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict = st.button("🔮 Predict Crop", use_container_width=True)

# Make prediction
if predict:
    with st.spinner("Analyzing..."):
        # Create input dataframe
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Predict
        prediction = model.predict(input_data)
        crop = encoder.inverse_transform(prediction)[0]
        
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
        
        # Simple recommendations
        st.subheader("💡 Quick Tips")
        if ph < 6.0:
            st.warning("Soil is acidic. Consider adding lime.")
        elif ph > 7.5:
            st.warning("Soil is alkaline. Consider adding sulfur.")
        else:
            st.success("Soil pH is optimal!")
        
        if N < 50:
            st.info("Low nitrogen. Add organic fertilizer.")