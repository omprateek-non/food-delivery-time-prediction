import streamlit as st
import pickle

# Load the trained model
model_file = 'trained_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict_delivery_time(features):
    return model.predict([features])

# Streamlit app layout
st.title('Food Delivery Time Prediction')

# User inputs
feature1 = st.number_input('Feature 1:')
feature2 = st.number_input('Feature 2:')
# Add more feature inputs as necessary

if st.button('Predict'):
    features = [feature1, feature2]  # Modify based on actual number of features
    prediction = predict_delivery_time(features)
    st.write(f'Estimated delivery time: {prediction[0]} minutes')
