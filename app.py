import streamlit as st
import pandas as pd
import pickle

# Load model and encoder
model = pickle.load(open("crop_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Title
st.title("🌱 Smart Crop Recommendation System")

st.write("Enter soil and weather details to get the best crop recommendation.")

# Inputs
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)

temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("pH Value", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Prediction Button
if st.button("Predict Crop"):

    sample = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=[
            'N',
            'P',
            'K',
            'temperature',
            'humidity',
            'ph',
            'rainfall'
        ]
    )

    prediction = model.predict(sample)

    crop = encoder.inverse_transform(prediction)

    st.success(f"Recommended Crop: {crop[0].upper()}")