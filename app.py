import os
import traceback
import pickle
import joblib
import streamlit as st
import pandas as pd
import numpy as np

MODEL_PATH = "25RP18587.pkl"

# -----------------------------
# Encoding mappings
# -----------------------------
region_map = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
soil_map   = {'Chalky': 0, 'Clay': 1, 'Loam': 2, 'Peaty': 3, 'Sandy': 4, 'Silt': 5}
crop_map   = {'Barley': 0, 'Cotton': 1, 'Maize': 2, 'Rice': 3, 'Soybean': 4, 'Wheat': 5}
fert_map   = {'No': 0, 'Yes': 1}
irrig_map  = {'No': 0, 'Yes': 1}
weather_map= {'Cloudy': 0, 'Rainy': 1, 'Sunny': 2}

# -----------------------------
# Safe model loader with fallbacks
# -----------------------------
def try_load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {os.path.abspath(path)}")
    # try pickle
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            pass
    # try joblib
    try:
        return joblib.load(path)
    except Exception:
        pass
    # try cloudpickle
    try:
        import cloudpickle
        with open(path, "rb") as f:
            return cloudpickle.load(f)
    except Exception:
        pass
    # If all failed, raise to allow caller to decide fallback
    raise RuntimeError("All attempts to load model failed (pickle/joblib/cloudpickle)")

# -----------------------------
# Fallback predictor (simple & explainable)
# -----------------------------
def fallback_predict(df):
    # A simple heuristic model: base yield + contributions from features
    # This is NOT a real model — it's for UI/demo purposes only.
    base = 2.0  # tons/ha base
    # weight coefficients (arbitrary for demo)
    w_rain = 0.01
    w_temp = 0.02
    w_days = 0.005
    w_fert = 0.8
    w_irrig = 0.9
    w_crop = 0.1
    w_soil = 0.05
    w_weather = 0.2

    preds = []
    for _, row in df.iterrows():
        val = base
        val += w_rain * row.get('Rainfall_mm', 0)
        # temp effect: optimal ~25 degC; penalty for deviation
        val += w_temp * max(0, 25 - abs(row.get('Temperature_Celsius', 25) - 25))
        val += w_days * row.get('Days_to_Harvest', 60)
        val += w_fert * row.get('Fertilizer_Used', 0)
        val += w_irrig * row.get('Irrigation_Used', 0)
        val += w_crop * (row.get('Crop', 0) * 0.5)
        val += w_soil * (5 - row.get('Soil_Type', 2)) * 0.1
        val += w_weather * (2 - row.get('Weather_Condition', 2)) * 0.1
        # add tiny noise
        val += np.random.normal(0, 0.05)
        preds.append(val)
    return np.array(preds)

# -----------------------------
# Load model (or set fallback)
# -----------------------------
loaded_model = None
model_load_error = None
try:
    loaded_model = try_load_model(MODEL_PATH)
except Exception as e:
    model_load_error = traceback.format_exc()
    loaded_model = None

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.markdown("## 👨‍💻 designed by : ISHIMWE Sowari Solange ")
    st.title('🌾 Crop Yield Prediction Portal')

    if loaded_model is None:
        st.warning("Model file could not be loaded — the app will use a safe demo fallback predictor. "
                   "This lets the UI run while you fix or provide the real model file.")
        with st.expander("Model load error (for debugging)"):
            st.code(model_load_error or "No error traceback available.")

    # Dropdowns
    region            = st.selectbox('Region:',            list(region_map.keys()))
    soil_type         = st.selectbox('Soil Type:',         list(soil_map.keys()))
    crop_type         = st.selectbox('Crop Type:',         list(crop_map.keys()))
    fertilizer_used   = st.selectbox('Fertilizer Used:',   ['No', 'Yes'])
    irrigation_used   = st.selectbox('Irrigation Used:',   ['No', 'Yes'])
    weather_condition = st.selectbox('Weather Condition:', list(weather_map.keys()))

    # Numeric inputs
    rainfall     = st.number_input('Rainfall (mm):',             min_value=0.0,   step=0.1, value=100.0)
    temperature  = st.number_input('Temperature (°C):',          min_value=-10.0, max_value=50.0, step=0.1, value=25.0)
    days_to_harvest = st.number_input('Days to Harvest:',         min_value=1,     step=1,   value=60)

    # Predict button
    if st.button('Predict Yield'):
        # Encode categorical
        region_encoded     = region_map[region]
        soil_encoded       = soil_map[soil_type]
        crop_encoded       = crop_map[crop_type]
        fert_encoded       = fert_map[fertilizer_used]
        irrig_encoded      = irrig_map[irrigation_used]
        weather_encoded    = weather_map[weather_condition]

        # Prepare input
        input_data = pd.DataFrame([{
            'Region':               region_encoded,
            'Soil_Type':            soil_encoded,
            'Crop':                 crop_encoded,
            'Rainfall_mm':          float(rainfall),
            'Temperature_Celsius':  float(temperature),
            'Fertilizer_Used':      fert_encoded,
            'Irrigation_Used':      irrig_encoded,
            'Weather_Condition':    weather_encoded,
            'Days_to_Harvest':      int(days_to_harvest)
        }])

        # Prediction: use real model if available, otherwise fallback
        try:
            if loaded_model is not None:
                # try sklearn-like .predict
                predicted_yield = loaded_model.predict(input_data)
            else:
                predicted_yield = fallback_predict(input_data)
        except Exception as ex:
            st.error("Error when calling the model's predict method. Falling back to demo predictor.")
            st.exception(ex)
            predicted_yield = fallback_predict(input_data)

        st.success(f'🌱 The predicted crop yield is: {predicted_yield[0]:.2f} tons per hectare')

if __name__ == '__main__':
    main()
