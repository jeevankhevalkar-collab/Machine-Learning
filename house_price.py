
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os

st.title('House Price Prediction')

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler
model_path = os.path.join(BASE_DIR, "Housepredictionmodel.joblib")
scaler_path = os.path.join(BASE_DIR, "Housepricepredictionscaler.joblib")

# Check files
if not os.path.exists(model_path):
    st.error("Housepredictionmodel.joblib file not found!")
    st.stop()

if not os.path.exists(scaler_path):
    st.error("Housepricepredictionscaler.joblib file not found!")
    st.stop()

Model = joblib.load(model_path)
Scaler = joblib.load(scaler_path)

# User Input
Col1, Col2, Col3 = st.columns(3)

SquareFootage = Col1.number_input("Square Footage", 500, 5000)
NumBedrooms = Col2.number_input("Number of Bedrooms", 1, 5)
NumBathrooms = Col3.number_input("Number of Bathrooms", 1, 3)

YearBuilt = Col1.number_input("Year Built", 1950, 2026)
LotSize = Col2.number_input("Size of Lot", 0.5, 5.0)
GarageSize = Col3.number_input("Size of Garage", 1, 10)
NeighborhoodQuality = Col2.number_input("Neighborhood Quality", 1, 5)

# Create DataFrame
PredictData = pd.DataFrame({
    'Square_Footage': [SquareFootage],
    'Num_Bedrooms': [NumBedrooms],
    'Num_Bathrooms': [NumBathrooms],
    'Year_Built': [YearBuilt],
    'Lot_Size': [LotSize],
    'Garage_Size': [GarageSize],
    'Neighborhood_Quality': [NeighborhoodQuality]
})

st.write(PredictData)

# Scale Data
pred_data_s = Scaler.transform(PredictData)

# Prediction
if st.button("Predict House Price"):
    Prediction = Model.predict(pred_data_s)
    st.success(f"Predicted House Price: ₹{Prediction[0]:,.2f}")
