import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and features
model = pickle.load(open('house_price_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

# Title
st.title("🏡 House Price Prediction App")
st.write("Enter the property details below:")

# Create input fields dynamically
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Scale the input
scaled_input = scaler.transform(input_df)
# Predict
if st.button("Predict Price"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"💰 Estimated Property Price: ${prediction:,.2f}")
