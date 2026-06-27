import streamlit as st
import pandas as pd
import pickle
import json
import sqlite3
from datetime import datetime

# Load model and info
@st.cache_resource
def load_data():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_info.json", "r") as f:
        info = json.load(f)
    return model, info

# Initialize the database
@st.cache_resource
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            distance REAL,
            weather TEXT,
            traffic TEXT,
            time_of_day TEXT,
            vehicle TEXT,
            prep_time REAL,
            experience REAL,
            predicted_time REAL
        )
        """
    )
    conn.commit()
    conn.close()

# Save a prediction to the database
def save_prediction(distance, weather, traffic, time_of_day, vehicle, prep_time, experience, predicted_time):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (timestamp, distance, weather, traffic, time_of_day, vehicle, prep_time, experience, predicted_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), distance, weather, traffic, time_of_day, vehicle, prep_time, experience, predicted_time)
    )
    conn.commit()
    conn.close()

# Call init_db at the top of the script
init_db()

try:
    model, model_info = load_data()
except FileNotFoundError:
    st.error("Model files not found. Please ensure model.pkl and model_info.json exist.")
    st.stop()

st.title("Food Delivery Time Predictor 🍔🛵")

st.write("Enter the delivery details below to predict the estimated delivery time.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numeric Features")
        distance = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        prep_time = st.number_input("Preparation Time (min)", min_value=0, max_value=120, value=20)
        experience = st.number_input("Courier Experience (yrs)", min_value=0, max_value=20, value=2)
        
    with col2:
        st.subheader("Categorical Features")
        weather = st.selectbox("Weather", ["Clear", "Rainy", "Cloudy", "Snowy", "Windy"])
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High", "Very High"])
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Car"])
        
    submit_button = st.form_submit_button("Predict Delivery Time")

if submit_button:
    # Create input dataframe
    input_data = {
        "Distance_km": distance,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": experience
    }
    
    input_df = pd.DataFrame([input_data])
    
    try:
        # Predict
        prediction = model.predict(input_df)[0]
        
        st.success(f"Estimated Delivery Time: **{prediction:.1f} minutes** ⏱️")
        
        st.info("Input Features Summary:")
        st.json(input_data)
        
        # Save prediction to the database
        save_prediction(distance, weather, traffic, time_of_day, vehicle, prep_time, experience, prediction)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Add prediction history section
with st.expander("Prediction History"):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, distance, weather, traffic, time_of_day, vehicle, prep_time, experience, predicted_time FROM predictions ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    if rows:
        history_df = pd.DataFrame(rows, columns=[
            "Timestamp", "Distance (km)", "Weather", "Traffic Level", "Time of Day", "Vehicle Type", "Preparation Time (min)", "Courier Experience (yrs)", "Predicted Time (min)"
        ])
        st.dataframe(history_df)
    else:
        st.write("No prediction history available.")