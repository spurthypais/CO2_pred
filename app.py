import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
from netCDF4 import Dataset
import cdsapi
from datetime import date

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(layout="wide", page_title="CO2 Prediction 🌱", page_icon="🌍")

st.markdown(
    """
    <h1 style='text-align:center; color: darkgreen;'>🌍 CO2 Prediction Dashboard</h1>
    <p style='text-align:center; font-size:16px;'>
    Select a date and location to predict CO2 levels using ERA5 meteorological & vegetation data.
    </p>
    """, unsafe_allow_html=True
)

# ---------------------------
# Set up CDS API credentials (Streamlit Cloud)
# ---------------------------
cds_path = os.path.expanduser("~/.cdsapirc")
if not os.path.exists(cds_path):
    try:
        with open(cds_path, "w") as f:
            f.write(
                f"url: https://cds.climate.copernicus.eu/api/v2\n"
                f"key: {st.secrets['CDSAPI_UID']}:{st.secrets['CDSAPI_KEY']}\n"
            )
    except KeyError:
        st.error("⚠️ CDS API credentials not set in Streamlit secrets!")

client = cdsapi.Client()

# ---------------------------
# Load trained model + scaler
# ---------------------------
MODEL_PATH = "rf_pipeline.pkl"  # make sure the .pkl file is in repo root
model, scaler_target = None, None

if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        model = bundle.get("model", None)
        scaler_target = bundle.get("scaler_target", None)
        if model and scaler_target:
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model or scaler missing in bundle")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.error("⚠️ Model file not found. Please check path!")

# ---------------------------
# Variable mappings
# ---------------------------
VAR_MAP = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "surface_pressure": "sp",
    "high_vegetation_cover": "cvh",
    "low_vegetation_cover": "cvl",
    "leaf_area_index_high_vegetation": "lai_hv",
    "leaf_area_index_low_vegetation": "lai_lv",
}

# ---------------------------
# ERA5 download function
# ---------------------------
def download_era5(year, month, day, save_path, variables, prefix):
    nc_filename = f"{prefix}_{year}-{month:02d}-{day:02d}.nc"
    nc_path = os.path.join(save_path, nc_filename)
    if not os.path.exists(nc_path):
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": variables,
                "year": str(year),
                "month": f"{month:02d}",
                "day": f"{day:02d}",
                "time": ["13:00"],
                "format": "netcdf"
            }
        ).download(nc_path)
    return nc_path

# ---------------------------
# Extract nearest values
# ---------------------------
def extract_nearest_values(nc_path, lat, lon, requested_vars):
    values = {}
    with Dataset(nc_path, mode="r") as nc_file:
        lon_values = nc_file.variables["longitude"][:]
        lat_values = nc_file.variables["latitude"][:]
        lon_idx = np.abs(lon_values - lon).argmin()
        lat_idx = np.abs(lat_values - lat).argmin()
        for req_var in requested_vars:
            nc_var = VAR_MAP.get(req_var, req_var)
            if nc_var in nc_file.variables:
                values[nc_var] = float(nc_file.variables[nc_var][0, lat_idx, lon_idx])
            else:
                values[nc_var] = None
    return values

# ---------------------------
# UI: Date and Map Selection
# ---------------------------
st.markdown("---")
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("📅 Select a Date")
    selected_date = st.date_input(
        "Date:",
        value=None,
        min_value=date(1950, 1, 1),
        max_value=date.today()
    )

with col2:
    st.subheader("📍 Select Location")
    m = folium.Map(location=[20,0], zoom_start=2)
    if "last_clicked" in st.session_state:
        folium.Marker(
            list(st.session_state["last_clicked"]),
            tooltip="Selected Location",
            icon=folium.Icon(color="red", icon="map-marker")
        ).add_to(m)
    map_data = st_folium(m, width=700, height=450)
    if map_data and map_data["last_clicked"]:
        st.session_state["last_clicked"] = (
            map_data["last_clicked"]["lat"],
            map_data["last_clicked"]["lng"]
        )

# ---------------------------
# Prediction logic
# ---------------------------
if selected_date and "last_clicked" in st.session_state:
    save_dir = "era5_data"
    os.makedirs(save_dir, exist_ok=True)
    lat, lon = st.session_state["last_clicked"]

    try:
        # --- Download ERA5 data ---
        meteo_vars = ["10m_u_component_of_wind","10m_v_component_of_wind","2m_temperature","surface_pressure"]
        veg_vars = ["high_vegetation_cover","leaf_area_index_high_vegetation","leaf_area_index_low_vegetation","low_vegetation_cover"]
        nc_meteo = download_era5(selected_date.year, selected_date.month, selected_date.day, save_dir, meteo_vars, "meteo")
        nc_veg = download_era5(selected_date.year, selected_date.month, selected_date.day, save_dir, veg_vars, "veg")

        # --- Extract values ---
        values_meteo = extract_nearest_values(nc_meteo, lat, lon, meteo_vars)
        values_veg = extract_nearest_values(nc_veg, lat, lon, veg_vars)
        all_values = {**values_meteo, **values_veg}

        # --- Prepare DataFrame ---
        df = pd.DataFrame([{"date": selected_date, "latitude": lat, "longitude": lon, **all_values}])
        df["year"], df["month"], df["day"] = selected_date.year, selected_date.month, selected_date.day

        # --- Predict CO2 ---
        if model and scaler_target:
            predictors = ['latitude','longitude','year','month','day','sp','t2m','u10','v10','lai_hv','lai_lv','cvh','cvl']
            if all(col in df.columns for col in predictors):
                X_input = df[predictors]
                y_pred = model.predict(X_input)
                y_pred = scaler_target.inverse_transform(y_pred.reshape(-1,1)).ravel()
                df["CO2"] = y_pred

                # --- Display final table ---
                st.markdown("---")
                st.subheader("🌱 Predicted CO2")
                st.dataframe(df[["date","latitude","longitude","CO2"]], use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Could not fetch data or predict: {e}")
else:
    st.info("👉 Please select both a date and a location")
