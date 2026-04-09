import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and encoders
try:
    with open('optimized_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title('🍕 Food Delivery Time Prediction')
st.write('Enter delivery details to predict estimated delivery time')

# Create input columns
col1, col2 = st.columns(2)

with col1:
    distance = st.number_input('Distance (km)', min_value=0.0, max_value=50.0, value=5.0)
    prep_time = st.number_input('Preparation Time (min)', min_value=0, max_value=60, value=15)
    experience = st.number_input('Courier Experience (years)', min_value=0.0, max_value=20.0, value=2.0)

with col2:
    weather = st.selectbox('Weather', ['Clear', 'Rainy', 'Foggy', 'Snowy', 'Windy'])
    traffic = st.selectbox('Traffic Level', ['Low', 'Medium', 'High'])
    time_of_day = st.selectbox('Time of Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    vehicle = st.selectbox('Vehicle Type', ['Bike', 'Scooter', 'Car'])

# Make prediction
if st.button('Predict Delivery Time'):
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Distance_km': [distance],
            'Preparation_Time_min': [prep_time],
            'Courier_Experience_yrs': [experience],
            'Weather': [weather],
            'Traffic_Level': [traffic],
            'Time_of_Day': [time_of_day],
            'Vehicle_Type': [vehicle]
        })
        
        # Make one-hot encoded features match the model expectations
        prediction = model.predict(input_data)
        
        st.success(f'✅ Estimated Delivery Time: **{int(prediction[0])} minutes**')
        
    except Exception as e:
        st.error(f'Error making prediction: {e}')