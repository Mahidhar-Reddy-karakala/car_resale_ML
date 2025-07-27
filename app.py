import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk

# Load the model
model = pk.load(open('car_resale_model.pkl', 'rb')) 

# Optional: Enable warning for downcasting
# pd.set_option('future.no_silent_downcasting', True)

# Streamlit UI
st.title("Car Resale Value Prediction")
st.write("Enter the details of the car to predict its resale value.")

# Input fields
Brand = st.selectbox("Brand", options=[
    'Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra',
    'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi',
    'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
    'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'
])
year = st.slider("Year of Manufacture", min_value=1994, max_value=2025)
kmDriven = st.number_input("Km Driven:")
fuel = st.selectbox("Fuel Type:", options=['Diesel', 'Petrol', 'LPG', 'CNG'])
seller_type = st.selectbox("Seller Type:", options=['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission:", options=['Manual', 'Automatic'])
owner = st.selectbox("Owner:", options=[
    'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'
])
mileage = st.number_input("Mileage (kmpl):", min_value=0)
engine = st.number_input("Engine (cc):", min_value=0)
max_power = st.number_input("Max Power (bhp):", min_value=0)
seats = st.selectbox("Seats:", options=[2, 4, 5, 6, 7])

# Predict button
if st.button("Predict Resale Value"):
    # Prepare input dataframe (NOTE: model trained with column name 'name', not 'brand')
    input_data = pd.DataFrame({
        'name': [Brand],
        'year': [year],
        'km_driven': [kmDriven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    })

    # Encode categorical fields (without inplace to avoid warnings)
    input_data['name'] = input_data['name'].replace([
        'Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra',
        'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi',
        'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
        'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'
    ], list(range(1, 32)))

    input_data['fuel'] = input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4])
    input_data['seller_type'] = input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3])
    input_data['transmission'] = input_data['transmission'].replace(['Manual', 'Automatic'], [1, 2])
    input_data['owner'] = input_data['owner'].replace([
        'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'
    ], [1, 2, 3, 4, 5])

    # Optional: apply infer_objects to prevent downcasting warning
    # input_data = input_data.infer_objects(copy=False)

    # Predict resale value
    predicted_value = model.predict(input_data)

    # Show result
    st.success(f"The predicted resale value of the car is: ₹{predicted_value[0]:,.2f}")
