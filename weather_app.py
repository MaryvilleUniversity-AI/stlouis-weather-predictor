import streamlit as st
import numpy as np 
import pandas as pd 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime
import requests

# Constants
LAT = 38.6270 # St. Louis latitude
LON = -90.1994 # St. Louis longitude
MODEL_PATH = 'stlouis_temperature_predictor.keras'
SCALER_PATH = 'scaler.save'

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Helper function to fetch weather data from Open-Meteo
def fetch_weather_openmeteo(lat, lon, date: datetime.date):
    '''
    Fetch daily weather data (TMIN, TMAX, precipitation, and windspeed)
    from the Open-Meteo Archive API for a specific date.
    Returns: (tmin, tmax, prcp, wind)
    '''
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date.isoformat(),
        "end_date": date.isoformat(),
        "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,windspeed_10m_max",
        "timezone": "America/Chicago"
    }

    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()
    # main.temp_min, main.temp_max, wind.speed, rain.1h or snow.1h
    tmin = data['daily']['temperature_2m_min'][0]
    tmax = data['daily']['temperature_2m_max'][0]
    prcp = data['daily']['precipitation_sum'][0]
    wind = data['daily']['windspeed_10m_max'][0]

    return tmin, tmax, prcp, wind

# Streamlit UI
st.title("🌤️St. Louis Temperature Predictor")
st.markdown("""
This app predicts the **average daily temperature** in St. Louis
using historical weather data from **Open-Meteo** and your trained neural network.
""")

# Sidebar: Date Input
st.sidebar.header("Select Date")
selected_date = st.sidebar.date_input(
    "Choose a date",
    value=datetime.date.today(),
    min_value=datetime.date(2023, 1, 1),
    max_value=datetime.date.today()
)

# Fetch data
with st.spinner("Fetching weather data..."):
    try:
        tmin_c, tmax_c, prcp, awnd = fetch_weather_openmeteo(LAT, LON, selected_date)
    except Exception as e:
        st.error(f"Could not fetch weather data: {e}")
        st.stop()

# Display fetched weather info
st.subheader("Weather Data Used for Prediction")
st.write(f"**Date:** {selected_date.strftime('%B %d, %Y')}")
st.write(f"**TMIN:** {tmin_c:.2f} °C")
st.write(f"**TMAX:** {tmax_c:.2f} °C")
st.write(f"**Precipitation:** {prcp:.2f} mm")
st.write(f"**Wind Speed:** {awnd:.2f} m/s")

# Prepare feature for prediction
day_of_year = selected_date.timetuple().tm_yday
user_input = np.array([[tmin_c, tmax_c, prcp, awnd, day_of_year]], dtype='float32')

# Scale input
user_input_scaled = scaler.transform(user_input)

# Make predicition
predicted_temp_c = model.predict(user_input_scaled)[0][0]
predicted_temp_f = predicted_temp_c * 9/5 + 32

# Display Output
st.subheader("Predicted Average Temperature:")
st.write(f"{predicted_temp_f:.2f} °F ( {predicted_temp_c:.2f} °C)")

# Footer
st.caption("Data source: [Open-Meteo.com](https://open-meteo.com/) | Model trained on NOAA dataset")