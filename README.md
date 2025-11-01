# St. Louis Temperature Predictor
An interative **Streamlit web app** that predicts the **average daily temperature** in **St. Louis, Missouri** using a trained deep learning model.
The app automatically fetches recent or historical weather data from the **Open-Meteo API** and uses that data to estimate the day's average temperature.

## Features
* Slect any date from an interactive calendar
* Automatically fetches **TMIN**, **TMAX**, **precipitation**, and **wind speed**
* Predicts **average daily temperature** using a trained neural network model
* Displays results in both **Celsius** and **Fahrenheit**
* Model and scaler loaded directly from saved .keras and .save files
* Built with **TensorFlow**, **scikit-learn**, and **Streamlit**
