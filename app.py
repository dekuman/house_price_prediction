import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("California Housing Price Predictor")
st.markdown("Enter the housing features below to predict the price.")

# Feature Inputs
MedInc = st.number_input("Median Income (10k USD units)", min_value=0.0, value=3.0)
HouseAge = st.number_input("House Age", min_value=0.0, value=20.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
Population = st.number_input("Population", min_value=0.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0)

if st.button("Predict"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    st.success(f"Predicted House Price: ${prediction * 100000:.2f}")