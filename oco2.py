import streamlit as st
import cdsapi
import zipfile
import os
import pandas as pd

# --- Streamlit Page Setup ---
st.set_page_config(page_title="ERA5 Downloader", layout="centered")

st.title("📥 ERA5 Data Downloader")
st.markdown("Select a date to download ERA5 single-level + ERA5 vegetation data (NetCDF inside a ZIP). A CSV log will be updated.")

# --- Date Input ---
selected_date = st.date_input("Select Date")

# --- Download Button ---
if st.button("Download ERA5 Data"):
    with st.spinner("Requesting ERA5 + Vegetation data..."):

        # Extract date components
        year = f"{selected_date.year}"
        month = f"{selected_date.month:02d}"
        day = f"{selected_date.day:02d}"
        date_str = f"{year}-{month}-{day}"

        client = cdsapi.Client()

        # --- Dataset 1: ERA5 Single Levels (atmosphere hourly) ---
        single_level_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "surface_pressure",
            "total_precipitation"
        ]

        request_single = {
            "product_type": "reanalysis",
            "variable": single_level_vars,
            "year": year,
            "month": month,
            "day": day,
            "time": ["13:00"],
            "format": "netcdf"
        }

        nc_single = f"era5_single_{year}{month}{day}.nc"
        client.retrieve("reanalysis-era5-single-levels", request_single).download(nc_single)

        # --- Dataset 2: ERA5 Single Levels (vegetation hourly) ---
        vegetation_vars = [
            "high_vegetation_cover",
            "low_vegetation_cover",
            "leaf_area_index_high_vegetation",
            "leaf_area_index_low_vegetation",
            "type_of_high_vegetation",
            "type_of_low_vegetation"
        ]

        request_veg = {
            "product_type": "reanalysis",
            "variable": vegetation_vars,
            "year": year,
            "month": month,
            "day": day,
            "time": ["13:00"],
            "format": "netcdf"
        }

        nc_veg = f"era5_veg_{year}{month}{day}.nc"
        client.retrieve("reanalysis-era5-single-levels", request_veg).download(nc_veg)

        # --- Combine into ZIP ---
        zip_filename = f"era5_combined_{year}{month}{day}.zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(nc_single)
            zipf.write(nc_veg)

        # ✅ CSV logging
        csv_file = "download_log.csv"
        new_entry = pd.DataFrame([{
            "date": date_str,
            "variables": ", ".join(single_level_vars + vegetation_vars),
            "netcdf_files": f"{nc_single}, {nc_veg}",
            "zip_file": zip_filename
        }])

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, new_entry], ignore_index=True)
        else:
            df = new_entry

        df.to_csv(csv_file, index=False)

        # Success message
        st.success(f"Downloaded and zipped: {zip_filename}")

        # Download button for the ZIP file
        with open(zip_filename, "rb") as f:
            st.download_button(
                "⬇️ Download ZIP File",
                f,
                file_name=zip_filename,
                mime="application/zip"
            )

        # Download button for the CSV log
        with open(csv_file, "rb") as f:
            st.download_button(
                "📑 Download CSV Log",
                f,
                file_name=csv_file,
                mime="text/csv"
            )

        # Cleanup temporary files
        os.remove(nc_single)
        os.remove(nc_veg)
        os.remove(zip_filename)
