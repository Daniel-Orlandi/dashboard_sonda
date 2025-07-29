# Create a new file: dashboard.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import glob
import re
import numpy as np
import pvlib
from pvlib.location import Location

# Set page config for crisis room use
st.set_page_config(
    page_title="Solar Data Monitor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for crisis room design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .station-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .subsection-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e9ecef;
    }
    .variable-selector {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Solar Data Monitor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time Solar Radiation & Meteorological Data</p>', unsafe_allow_html=True)


def calculate_clearsky_ineichen(df, latitude, longitude, altitude=0, tz='UTC'):
    """
    Adiciona ao DataFrame estimativas de irradiância de céu limpo (GHI, DNI, DHI)
    usando o modelo Ineichen da biblioteca pvlib.
    """
    
    df = df.copy()
    
    # Garante datetime com timezone
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df = df.dropna(subset=['TIMESTAMP'])

    # Localiza ou converte o timezone
    if df['TIMESTAMP'].dt.tz is None:
        df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_localize(tz)
    else:
        df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_convert(tz)

    # Usa DatetimeIndex
    df = df.set_index('TIMESTAMP')

    # Cria o objeto Location
    location = pvlib.location.Location(latitude, longitude, tz=tz, altitude=altitude)

    # Calcula céu limpo com DatetimeIndex
    clearsky = location.get_clearsky(df.index)

    # Adiciona ao DataFrame
    df['clearsky_GHI'] = clearsky['ghi']
    df['clearsky_DNI'] = clearsky['dni']
    df['clearsky_DHI'] = clearsky['dhi']

    return df.reset_index()


# Function to get available stations
@st.cache_data
def get_available_stations():
    """Get list of available stations from the interim directory"""
    interim_dir = os.path.expanduser("data/interim")
    
    if not os.path.exists(interim_dir):
        return []
    
    # Normaliza os nomes das pastas das estações
    station_dirs = [d.strip().lower() for d in os.listdir(interim_dir) if os.path.isdir(os.path.join(interim_dir, d))]
    return sorted(station_dirs)


# Function to get available metadata
@st.cache_data
def get_station_metadata(station_dirs):
    """Get metadata of available stations (latitude and longitude), filtered by actual station folders"""
    interim_dir = os.path.expanduser("data/interim")
    location_csv = os.path.join(interim_dir, 'INPESONDA_Stations.csv')

    if not os.path.exists(interim_dir) or not os.path.exists(location_csv):
        st.warning("Arquivo de localização 'INPESONDA_Stations.csv' não encontrado.")
        return pd.DataFrame(columns=['station', 'latitude', 'longitude'])

    # Lê e normaliza os nomes das estações do CSV
    df_locations = pd.read_csv(location_csv)
    df_locations['station_normalized'] = df_locations['station'].astype(str).str.strip().str.lower()

    # Filtra as estações presentes nas pastas
    df_filtered = df_locations[df_locations['station_normalized'].isin(station_dirs)].copy()
    df_filtered = df_filtered.drop_duplicates(subset='station_normalized')

    return df_filtered.sort_values('station').reset_index(drop=True)


# Function to get the latest file for a specific station and data type
@st.cache_data
def get_latest_files_for_station(station):
    """Get the latest parquet file for a specific station and data type (SD, MD, WD)"""
    interim_dir = os.path.expanduser("data/interim")
    station_path = os.path.join(interim_dir, station)
    
    if not os.path.exists(station_path):
        return {}
    
    files = glob.glob(os.path.join(station_path, "*.parquet"))
    
    # Group files by data type (SD, MD, WD)
    files_by_type = {}
    for file in files:
        filename = os.path.basename(file)
        # Extract data type from filename (e.g., processed_data_PTR_SD_20250702_193759.parquet)
        match = re.search(r'_([A-Z]{2})_\d{8}_\d{6}\.parquet$', filename)
        if match:
            data_type = match.group(1)
            if data_type not in files_by_type:
                files_by_type[data_type] = []
            files_by_type[data_type].append(file)
    
    # Get the latest file for each data type
    latest_files = {}
    for data_type, file_list in files_by_type.items():
        if file_list:
            # Sort by modification time and get the latest
            latest_file = max(file_list, key=os.path.getmtime)
            latest_files[data_type] = latest_file
    
    return latest_files

def safe_convert_timestamp(df, timestamp_col='TIMESTAMP'):
    """
    Safely convert timestamp column to datetime, handling various formats and issues
    """
    if timestamp_col not in df.columns:
        return df
    
    # Check if already datetime
    if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        return df
    
    # Get sample of timestamp values to understand the format
    sample_values = df[timestamp_col].dropna().head(10).astype(str)
    
    # Try to identify the format
    for sample in sample_values:
        if pd.isna(sample) or sample == '':
            continue
            
        # Check if it's already a valid datetime string
        try:
            pd.to_datetime(sample)
            # If successful, try to convert the whole column
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                return df
            except Exception:
                break
        except:
            continue
    
    # If we get here, the timestamps might be in a different format
    # Check if the values look like they might be numeric timestamps
    sample_values = df[timestamp_col].dropna().head(5).astype(str)
    numeric_like = all(val.replace('.', '').replace('-', '').isdigit() for val in sample_values if val)
    
    if numeric_like:
        # Create synthetic timestamps (assuming 1-minute intervals)
        start_time = datetime.now() - timedelta(minutes=len(df))
        timestamps = [start_time + timedelta(minutes=i) for i in range(len(df))]
        df[timestamp_col] = timestamps
    else:
        # Create synthetic timestamps (assuming 1-minute intervals)
        start_time = datetime.now() - timedelta(minutes=len(df))
        timestamps = [start_time + timedelta(minutes=i) for i in range(len(df))]
        df[timestamp_col] = timestamps
    
    return df

def clean_numeric_columns(df):
    """
    Clean numeric columns by converting to numeric and handling invalid values
    """
    for col in df.columns:
        if col in ['TIMESTAMP', 'source_file', 'station', 'data_type', 'file_path']:
            continue
            
        # Try to convert to numeric, handling various formats
        try:
            # First try direct conversion
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            # If that fails, try to clean the data first
            try:
                # Remove common non-numeric characters and try again
                cleaned_values = df[col].astype(str).str.replace('NAN', 'NaN', case=False)
                cleaned_values = cleaned_values.str.replace('"', '')
                cleaned_values = cleaned_values.str.replace("'", '')
                df[col] = pd.to_numeric(cleaned_values, errors='coerce')
            except:
                # If all else fails, keep as is
                continue
    
    return df

# Function to load data from latest files for a specific station
@st.cache_data
def load_latest_data_for_station(station):
    """Load data from the latest files for a specific station and data type"""
    latest_files = get_latest_files_for_station(station)
    
    if not latest_files:
        return None
    
    data_dict = {}
    
    for data_type, file_path in latest_files.items():
        try:
            df = pd.read_parquet(file_path)
            print("Colunas disponíveis no arquivo:", df.columns.tolist())
            
            # Clean the data
            df = clean_numeric_columns(df)
            df = safe_convert_timestamp(df)
            
            # Add metadata columns
            df['source_file'] = os.path.basename(file_path)
            df['station'] = station
            df['data_type'] = data_type
            df['file_path'] = file_path
            
            data_dict[data_type] = df
            
        except Exception:
            continue
    
    return data_dict

def get_available_variables(df):
    """Get available numeric variables from a DataFrame, excluding date/time related columns"""
    # Get numeric columns only (excluding metadata columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove timestamp-related columns, metadata, and date/time related columns
    exclude_cols = ['TIMESTAMP', 'source_file', 'station', 'data_type', 'file_path', 
                   'Id', 'Min', 'RECORD', 'Year', 'Jday']
    available_vars = [col for col in numeric_cols if col not in exclude_cols]
    return available_vars

def create_variable_selector(available_vars, default_vars, key_prefix):
    """Create a multi-select widget for variables with default selection"""
    selected_vars = st.multiselect(
        "Select variables to display:",
        options=available_vars,
        default=default_vars,
        key=f"{key_prefix}_selector"
    )
    return selected_vars

def plot_selected_variables(df, selected_vars, plot_title, height=300):
    """Create a line plot for selected variables"""
    if not selected_vars or 'TIMESTAMP' not in df.columns:
        return
    
    try:
        plot_data = df[['TIMESTAMP'] + selected_vars].set_index('TIMESTAMP')
        st.line_chart(plot_data, height=height, use_container_width=True)
        st.caption(f"{plot_title}: {', '.join(selected_vars)}")
    except Exception as e:
        st.error(f"Error plotting {plot_title}: {str(e)}")


# Primeiro, obtém a lista de pastas das estações disponíveis
available_stations = get_available_stations()

# Depois, passa essa lista como argumento para a função de metadata
station_metadata = get_station_metadata(available_stations)      

if not available_stations:
    print("No stations found in the directory 'data/interim'.")
else:
    print("Available stations:", available_stations)

    
# Station selector - centered and prominent
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_station = st.selectbox(
        "Select Station:",
        options=available_stations,
        index=0,
        format_func=lambda x: x.upper()
    )

st.markdown(f'<h2 class="station-header">Station: {selected_station.upper()}</h2>', unsafe_allow_html=True)

# Load data for selected station
data_dict = load_latest_data_for_station(selected_station)

# Data Overview Section - Compact and informative
if data_dict is not None:
    st.header("📊 Data Overview")

    # Add legend for metrics
    st.markdown("""
<div style="font-size:1rem; color:#444; margin-bottom:0.5rem;">
<b>Legend:</b> <b>Records</b> = number of data rows; <b>Columns</b> = number of variables; <b>Updated</b> = last file modification time
</div>
""", unsafe_allow_html=True)

    # Display information about loaded files in a compact format
    file_info = []
    for data_type, df in data_dict.items():
        timestamp_col = 'TIMESTAMP'
        time_range = "N/A"
        if timestamp_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            time_range = f"{df[timestamp_col].min().strftime('%Y-%m-%d %H:%M')} to {df[timestamp_col].max().strftime('%Y-%m-%d %H:%M')}"
        file_info.append({
            'Data Type': data_type,
            'Records': len(df),
            'Columns': len(df.columns),
            'Time Range': time_range,
            'Last Modified': datetime.fromtimestamp(os.path.getmtime(df['file_path'].iloc[0])).strftime('%Y-%m-%d %H:%M')
        })
    file_df = pd.DataFrame(file_info)
    # Display in columns for better space usage
    cols = st.columns(len(file_info))
    for i, (_, row) in enumerate(file_df.iterrows()):
        with cols[i]:
            st.metric(
                label=f"{row['Data Type']} Data",
                value=f"{row['Records']:,}",
                delta=f"{row['Columns']} cols"
            )
            st.caption(f"Updated: {row['Last Modified']}")

    # Data Visualization Section - Three column layout
    st.header("📈 Real-time Data Visualization")
    col1, col2, col3 = st.columns([1, 1, 1])

    # Column 1: Solar Data (SD)
    with col1:
        st.subheader("☀️ Solar Data (SD)")
        
        if 'SD' in data_dict:
            df = data_dict['SD'].copy()
            
            # Get station metadata for clear sky calculation
            filtered_row = station_metadata[
                           station_metadata['station'].str.lower() == selected_station.lower()]

            if not filtered_row.empty:
                latitude = filtered_row['latitude'].values[0]
                longitude = filtered_row['longitude'].values[0]
                
                # Apply clear sky model to add clear sky variables
                df = calculate_clearsky_ineichen(df, latitude, longitude, tz='America/Sao_Paulo')
            
            available_vars = get_available_variables(df)
            
            if available_vars and 'TIMESTAMP' in df.columns:
                # Solar Radiation Plot (including clear sky data)
                st.markdown('<p class="subsection-header">Solar Radiation</p>', unsafe_allow_html=True)
                solar_vars = [var for var in available_vars if any(x in var.lower() for x in ['glo', 'dir', 'dif']) and any(x in var.lower() for x in ['avg', 'std'])]
                
                # Add clear sky variables if they exist
                clearsky_vars = ['clearsky_GHI', 'clearsky_DNI', 'clearsky_DHI']
                available_clearsky_vars = [var for var in clearsky_vars if var in df.columns]
                
                if solar_vars or available_clearsky_vars:
                    # Combine solar and clear sky variables
                    all_solar_vars = solar_vars + available_clearsky_vars
                    selected_solar = create_variable_selector(all_solar_vars, all_solar_vars, "solar")
                    plot_selected_variables(df, selected_solar, "Solar Radiation")
                else:
                    st.info("No solar radiation variables found.")
                
                # Longwave Radiation Plot
                st.markdown('<p class="subsection-header">Longwave Radiation</p>', unsafe_allow_html=True)
                lw_vars = [var for var in available_vars if 'lw' in var.lower() and any(x in var.lower() for x in ['avg', 'std'])]
                if lw_vars:
                    selected_lw = create_variable_selector(lw_vars, lw_vars, "longwave")
                    plot_selected_variables(df, selected_lw, "Longwave Radiation")
                else:
                    st.info("No longwave radiation variables found.")
                
                # PAR & LUX Plot
                st.markdown('<p class="subsection-header">PAR & LUX</p>', unsafe_allow_html=True)
                par_lux_vars = [var for var in available_vars if any(x in var.lower() for x in ['par', 'lux']) and any(x in var.lower() for x in ['avg', 'std'])]
                if par_lux_vars:
                    selected_par_lux = create_variable_selector(par_lux_vars, par_lux_vars, "par_lux")
                    plot_selected_variables(df, selected_par_lux, "PAR & LUX")
                else:
                    st.info("No PAR & LUX variables found.")
                
                # Temperature Plot
                st.markdown('<p class="subsection-header">Temperatures</p>', unsafe_allow_html=True)
                temp_vars = [var for var in available_vars if any(x in var.lower() for x in ['tp_', 'temp']) and any(x in var.lower() for x in ['avg', 'std'])]
                if temp_vars:
                    selected_temp = create_variable_selector(temp_vars, temp_vars, "temperature")
                    plot_selected_variables(df, selected_temp, "Temperatures")
                else:
                    st.info("No temperature variables found.")
            else:
                st.warning("No numeric data available for SD.")
        else:
            st.info("No SD data available for this station.")

    # Column 2: Environmental Data (MD and WD)
    with col2:
        st.subheader("🌤️ Environmental Data")
        
        # Meteorological Data Subsection
        st.markdown('<p class="subsection-header">Meteorological Data</p>', unsafe_allow_html=True)
        if 'MD' in data_dict:
            df_md = data_dict['MD'].copy()
            available_vars_md = get_available_variables(df_md)
            
            if available_vars_md and 'TIMESTAMP' in df_md.columns:
                meteo_vars = [var for var in available_vars_md if any(x in var.lower() for x in ['tp_sfc', 'humid', 'press', 'rain'])]
                if meteo_vars:
                    selected_meteo = create_variable_selector(meteo_vars, meteo_vars, "meteorological")
                    plot_selected_variables(df_md, selected_meteo, "Meteorological Data")
                else:
                    st.info("No meteorological variables found in MD data.")
            else:
                st.warning("No numeric data available for MD.")
        else:
            st.info("No MD data available for this station.")
        
        # Wind Data Subsection
        st.markdown('<p class="subsection-header">Wind Data</p>', unsafe_allow_html=True)
        
        # Wind at different heights
        wind_heights = ['10m', '25m', '50m']
        for height in wind_heights:
            wind_vars = []
            df_source = None
            
            # Check both MD and WD for wind data
            for data_type in ['MD', 'WD']:
                if data_type in data_dict:
                    df_temp = data_dict[data_type].copy()
                    available_vars_temp = get_available_variables(df_temp)
                    height_vars = [var for var in available_vars_temp if height in var]
                    if height_vars:
                        wind_vars = height_vars
                        df_source = df_temp
                        break
            
            if wind_vars and df_source is not None:
                st.write(f"**Wind at {height}**")
                selected_wind = create_variable_selector(wind_vars, wind_vars, f"wind_{height}")
                plot_selected_variables(df_source, selected_wind, f"Wind at {height}", height=200)
            else:
                st.info(f"No wind data available at {height}.")
        
        # Temperature per Level Subsection
        st.markdown('<p class="subsection-header">Temperature per Level</p>', unsafe_allow_html=True)
        if 'WD' in data_dict:
            df_wd = data_dict['WD'].copy()
            available_vars_wd = get_available_variables(df_wd)
            
            if available_vars_wd and 'TIMESTAMP' in df_wd.columns:
                temp_humid_vars = [var for var in available_vars_wd if any(x in var.lower() for x in ['tp_', 'humid_'])]
                if temp_humid_vars:
                    selected_temp_humid = create_variable_selector(temp_humid_vars, temp_humid_vars, "temp_humid_levels")
                    plot_selected_variables(df_wd, selected_temp_humid, "Temperature & Humidity per Level")
                else:
                    st.info("No temperature/humidity variables found in WD data.")
            else:
                st.warning("No numeric data available for WD.")
        else:
            st.info("No WD data available for this station.")

    # Column 3: Detailed View
    with col3:
        st.subheader("🔍 Detailed View")
        
        # Collect only variables that are shown in other sections
        detailed_variables = []
        
        # Get SD variables (Solar Data section)
        if 'SD' in data_dict:
            df_sd = data_dict['SD'].copy()
            
            # Get station metadata for clear sky calculation
            filtered_row = station_metadata[
                           station_metadata['station'].str.lower() == selected_station.lower()]

            if not filtered_row.empty:
                latitude = filtered_row['latitude'].values[0]
                longitude = filtered_row['longitude'].values[0]
                
                # Apply clear sky model to add clear sky variables
                df_sd = calculate_clearsky_ineichen(df_sd, latitude, longitude, tz='America/Sao_Paulo')
            else:
                st.warning(f"A estação '{selected_station.lower()}' não foi encontrada no metadata.")
                st.stop()
            
            available_vars_sd = get_available_variables(df_sd)
            
            # Solar radiation variables (glo, dir, dif with avg/std)
            solar_vars = [var for var in available_vars_sd if any(x in var.lower() for x in ['glo', 'dir', 'dif']) and any(x in var.lower() for x in ['avg', 'std'])]
            detailed_variables.extend([f"SD: {var}" for var in solar_vars])
            
            # Clear sky variables
            clearsky_vars = ['clearsky_GHI', 'clearsky_DNI', 'clearsky_DHI']
            available_clearsky_vars = [var for var in clearsky_vars if var in df_sd.columns]
            detailed_variables.extend([f"SD: {var}" for var in available_clearsky_vars])
            
            # Longwave variables (lw with avg/std)
            lw_vars = [var for var in available_vars_sd if 'lw' in var.lower() and any(x in var.lower() for x in ['avg', 'std'])]
            detailed_variables.extend([f"SD: {var}" for var in lw_vars])
            
            # PAR & LUX variables (par, lux with avg/std)
            par_lux_vars = [var for var in available_vars_sd if any(x in var.lower() for x in ['par', 'lux']) and any(x in var.lower() for x in ['avg', 'std'])]
            detailed_variables.extend([f"SD: {var}" for var in par_lux_vars])
            
            # Temperature variables (tp_, temp with avg/std)
            temp_vars = [var for var in available_vars_sd if any(x in var.lower() for x in ['tp_', 'temp']) and any(x in var.lower() for x in ['avg', 'std'])]
            detailed_variables.extend([f"SD: {var}" for var in temp_vars])
        
        # Get MD variables (Environmental Data section)
        if 'MD' in data_dict:
            df_md = data_dict['MD'].copy()
            available_vars_md = get_available_variables(df_md)
            
            # Meteorological variables
            meteo_vars = [var for var in available_vars_md if any(x in var.lower() for x in ['tp_sfc', 'humid', 'press', 'rain'])]
            detailed_variables.extend([f"MD: {var}" for var in meteo_vars])
            
            # Wind variables at different heights
            wind_heights = ['10m', '25m', '50m']
            for height in wind_heights:
                height_vars = [var for var in available_vars_md if height in var]
                detailed_variables.extend([f"MD: {var}" for var in height_vars])
        
        # Get WD variables (Environmental Data section)
        if 'WD' in data_dict:
            df_wd = data_dict['WD'].copy()
            available_vars_wd = get_available_variables(df_wd)
            
            # Wind variables at different heights
            wind_heights = ['10m', '25m', '50m']
            for height in wind_heights:
                height_vars = [var for var in available_vars_wd if height in var]
                detailed_variables.extend([f"WD: {var}" for var in height_vars])
            
            # Temperature and humidity per level
            temp_humid_vars = [var for var in available_vars_wd if any(x in var.lower() for x in ['tp_', 'humid_'])]
            detailed_variables.extend([f"WD: {var}" for var in temp_humid_vars])
        
        if detailed_variables:
            # Variable selector for detailed view
            st.write("**Select variable for detailed view:**")
            
            selected_detailed_var = st.selectbox(
                "Choose variable:",
                options=detailed_variables,
                key="detailed_var_selector"
            )
            
            if selected_detailed_var:
                data_type, var_name = selected_detailed_var.split(": ", 1)
                df_detail = data_dict[data_type].copy()
                
                if 'TIMESTAMP' in df_detail.columns and var_name in df_detail.columns:
                    st.write(f"**{var_name} ({data_type})**")
                    
                    # Create detailed plot
                    plot_data = df_detail[['TIMESTAMP', var_name]].set_index('TIMESTAMP')
                    st.line_chart(plot_data, height=400, use_container_width=True)
                    
                    # Show statistics
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Mean", f"{df_detail[var_name].mean():.2f}")
                        st.metric("Min", f"{df_detail[var_name].min():.2f}")
                    with col_stat2:
                        st.metric("Max", f"{df_detail[var_name].max():.2f}")
                        st.metric("Std Dev", f"{df_detail[var_name].std():.2f}")
                else:
                    st.error(f"Variable {var_name} not found in {data_type} data.")
        else:
            st.info("No variables available for detailed view.")

    # Raw Data Section (collapsible) - Only show if needed
    with st.expander("📋 Raw Data (Click to expand)"):
        for data_type, df in data_dict.items():
            st.write(f"**{selected_station.upper()} - {data_type} Data**")
            st.write(f"Shape: {df.shape}")
            st.dataframe(df.head(20), use_container_width=True)  # Show first 20 rows only

else:
    st.error("No data files found for the selected station.")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #666; font-size: 0.9rem;">Solar Data Monitoring System - Crisis Room Dashboard</p>', unsafe_allow_html=True)