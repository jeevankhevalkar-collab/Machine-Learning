import pandas as pd  # noqa: I001
import numpy as np  # noqa: F401
import streamlit as st

import joblib



st.title('House Price Prediction')

Model = joblib.load('Housepredictionmodel.joblib')
Scaler = joblib.load('Housepricepredictionscaler.joblib')

Col1, Col2, Col3 = st.columns(3)
SquareFootage = Col1.number_input("Square Footage", 500, 5000)
NumBedrooms = Col2.number_input("Number of Bedrooms", 1, 5)
NumBathrooms = Col3.number_input("Number of Bathrooms", 1, 3)
YearBuilt = Col1.number_input("Year Built", 1950, 2026)
LotSize = Col2.number_input("Size of Lot", 0.5, 5.0)
GarageSize = Col3.number_input("Size of Garage", 1, 10)
NeighborhoodQuality = Col2.number_input("Neighborhood Quality", 1, 5)

PredictData = pd.DataFrame({
    'Square_Footage' : [SquareFootage],
    'Num_Bedrooms' : [NumBedrooms],
    'Num_Bathrooms' : [NumBathrooms],
    'Year_Built' : [YearBuilt],
    'Lot_Size' : [LotSize],
    'Garage_Size' : [GarageSize],
    'Neighborhood_Quality' : [NeighborhoodQuality]
})
st.write(PredictData)

pred_data_s=Scaler.transform(PredictData)

PredictButton = st.button("Predict Price")
if PredictButton:
    Price = Model.predict(PredictData)
    st.title(f'${round(Price[0], 0)}')
