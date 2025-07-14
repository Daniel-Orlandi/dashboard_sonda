# Create a new file: dashboard.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import glob
import psycopg2
from sqlalchemy import create_engine
import re

# Set page config
st.set_page_config(
    page_title="Solar Data Pipeline Monitor",
    page_icon="☀️",
    layout="wide"
)
# lado direito pra um olhar mais detalhado, colocar seletor de estações (vini)
# Title
st.title("Solar Data Pipeline Monitor")
st.markdown("**Data Operator Dashboard for FTP Download Pipeline Monitoring**")

# Function to get available stations
def get_available_stations():
    """Get list of available stations from the interim directory"""
    interim_dir = os.path.expanduser("data/interim")
    
    if not os.path.exists(interim_dir):
        return []
    
    # Look for station subdirectories
    station_dirs = [d for d in os.listdir(interim_dir) if os.path.isdir(os.path.join(interim_dir, d))]
    return sorted(station_dirs)

# Function to get the latest file for a specific station and data type
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

# Function to load data from latest files for a specific station
def load_latest_data_for_station(station):
    """Load data from the latest files for a specific station and data type"""
    latest_files = get_latest_files_for_station(station)
    
    if not latest_files:
        st.error(f"No data files found for station {station}")
        return None
    
    data_dict = {}
    
    for data_type, file_path in latest_files.items():
        try:
            df = pd.read_parquet(file_path)
            # Add metadata columns
            df['source_file'] = os.path.basename(file_path)
            df['station'] = station
            df['data_type'] = data_type
            df['file_path'] = file_path
            data_dict[data_type] = df
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
    
    return data_dict

# Get available stations
available_stations = get_available_stations()

if not available_stations:
    st.error("No stations found in the interim directory. Please check the data pipeline and ensure files are being generated.")
    st.stop()

# Station selector
st.header("Station Selection")
selected_station = st.selectbox(
    "Select Station:",
    options=available_stations,
    index=0,  # Default to first station
    format_func=lambda x: x.upper()  # Display station names in uppercase
)

st.markdown(f"**Currently viewing data for station: {selected_station.upper()}**")

# Load data for selected station
data_dict = load_latest_data_for_station(selected_station)

# Add this function to get DAG status
def get_dag_status():
    try:
        # Connect to PostgreSQL directly
        conn = psycopg2.connect(
            dbname="airflow",
            user="airflow",
            password="airflow",
            host="postgres",
            port="5432"
        )
        
        # Query to get the latest DAG runs and their status
        query = """
        SELECT 
            dag_id,
            run_id,
            state,
            start_date,
            end_date,
            logical_date,
            run_type,
            queued_at
        FROM dag_run
        WHERE dag_id LIKE '%solar%' OR dag_id LIKE '%data%'
        ORDER BY start_date DESC
        LIMIT 10
        """
        
        # Read into DataFrame
        dag_status = pd.read_sql_query(query, conn)
        conn.close()
        
        if dag_status.empty:
            st.warning("""
            No DAG runs found in the database. Please:
            1. Go to Airflow UI (http://localhost:8080)
            2. Enable your DAGs
            3. Trigger a run manually or wait for scheduled runs
            """)
            return None
            
        return dag_status
    except Exception as e:
        st.error(f"Error fetching DAG status: {str(e)}")
        return None

# Define variable groups for each data type
def get_variable_groups():
    """Define which variables belong to which data type and plot group"""
    return {
        'SD': {
            'Solar Radiation': ['glo_avg', 'dir_avg', 'dif_avg', 'glo_std', 'dir_std', 'dif_std'],
            'Longwave Radiation': ['lw_raw_avg', 'lw_calc_avg', 'lw_raw_std', 'lw_calc_std'],
            'PAR & LUX': ['par_avg', 'lux_avg', 'par_std', 'lux_std'],
            'Temperatures': ['tp_dir', 'tp_lw_case'],
            'Tilt': ['tilt_avg', 'tilt_std']
        },
        'MD': {
            'Meteorological': ['tp_sfc', 'humid', 'press', 'rain'],
            'Wind 10m': ['ws10_avg', 'wd10_avg', 'ws10_std', 'wd10_std']
        },
        'WD': {
            'Wind 10m': ['ws10_avg', 'wd10_avg', 'ws10_std', 'wd10_std'],
            'Wind 25m': ['ws25_avg', 'wd25_avg', 'ws25_std', 'wd25_std'],
            'Wind 50m': ['ws50_avg', 'wd50_avg', 'ws50_std', 'wd50_std'],
            'Temperatures': ['tp_25', 'tp_50', 'humid_25', 'humid_50']
        }
    }

if data_dict is not None:
    # Pipeline Status Section
    st.header("Pipeline Status")
    
    # Get DAG status
    dag_status = get_dag_status()
    
    if dag_status is not None:
        # Create a styled DataFrame
        def color_status(val):
            color_map = {
                'success': 'background-color: #90EE90',  # Light green
                'failed': 'background-color: #FFB6C1',   # Light red
                'running': 'background-color: #ADD8E6',  # Light blue
                'queued': 'background-color: #D3D3D3',  # Light gray
                'scheduled': 'background-color: #FFE4B5'  # Light orange
            }
            return color_map.get(val.lower(), '')
        
        # Format the DataFrame
        display_df = dag_status[[
            'dag_id',
            'run_id',
            'state',
            'logical_date',
            'start_date',
            'end_date',
            'run_type'
        ]].copy()
        
        # Convert datetime columns
        for col in ['logical_date', 'start_date', 'end_date']:
            display_df[col] = pd.to_datetime(display_df[col])
        
        # Calculate duration
        display_df['duration'] = (display_df['end_date'] - display_df['start_date']).dt.total_seconds() / 60
        
        # Format column names for display
        display_df.columns = [
            'Pipeline',
            'Run ID',
            'Status',
            'Logical Date',
            'Start Time',
            'End Time',
            'Run Type',
            'Duration (min)'
        ]
        
        # Apply styling
        styled_df = display_df.style.applymap(
            color_status,
            subset=['Status']
        )
        
        # Display the styled table
        st.dataframe(styled_df)
        
        # Add status summary
        status_counts = dag_status['state'].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Runs", len(dag_status))
        with col2:
            st.metric("Successful", status_counts.get('success', 0))
        with col3:
            st.metric("Failed", status_counts.get('failed', 0))

    # Data Overview Section
    st.header("Latest Data Files")
    
    # Display information about loaded files
    file_info = []
    for data_type, df in data_dict.items():
        file_info.append({
            'Data Type': data_type,
            'File': os.path.basename(df['file_path'].iloc[0]),
            'Records': len(df),
            'Time Range': f"{df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}",
            'Last Modified': datetime.fromtimestamp(os.path.getmtime(df['file_path'].iloc[0]))
        })
    
    file_df = pd.DataFrame(file_info)
    st.dataframe(file_df, use_container_width=True)

    # Data Visualization Section
    st.header("Data Visualization")
    
    # Get variable groups
    variable_groups = get_variable_groups()
    
    # Create three columns layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # First column: Solar/Radiation Data
    with col1:
        st.subheader("Solar Data (SD)")
        if 'SD' in data_dict:
            sd_data = data_dict['SD'].copy()
            sd_data['TIMESTAMP'] = pd.to_datetime(sd_data['TIMESTAMP'])
            
            # Variable selection for SD
            st.write("**Select Variables to Plot:**")
            for group_name, variables in variable_groups['SD'].items():
                # Check which variables are actually available in the data
                available_vars = [var for var in variables if var in sd_data.columns]
                if available_vars:
                    selected_vars = st.multiselect(
                        f"{group_name}:",
                        options=available_vars,
                        default=available_vars,
                        key=f"sd_{group_name}"
                    )
                    
                    if selected_vars:
                        plot_data = sd_data[['TIMESTAMP'] + selected_vars].set_index('TIMESTAMP')
                        st.line_chart(plot_data, use_container_width=True)
        else:
            st.warning("No SD data available")
    
    # Second column: Environmental and Wind Data
    with col2:
        st.subheader("Environmental & Wind Data")
        
        # Environmental Data (MD)
        st.write("**Environmental Data (MD):**")
        if 'MD' in data_dict:
            md_data = data_dict['MD'].copy()
            md_data['TIMESTAMP'] = pd.to_datetime(md_data['TIMESTAMP'])
            
            # Variable selection for MD
            for group_name, variables in variable_groups['MD'].items():
                # Check which variables are actually available in the data
                available_vars = [var for var in variables if var in md_data.columns]
                if available_vars:
                    selected_vars = st.multiselect(
                        f"{group_name}:",
                        options=available_vars,
                        default=available_vars,
                        key=f"md_{group_name}"
                    )
                    
                    if selected_vars:
                        plot_data = md_data[['TIMESTAMP'] + selected_vars].set_index('TIMESTAMP')
                        st.line_chart(plot_data, use_container_width=True)
        else:
            st.warning("No MD data available")
        
        # Wind Data (WD)
        st.write("**Wind Data (WD):**")
        if 'WD' in data_dict:
            wd_data = data_dict['WD'].copy()
            wd_data['TIMESTAMP'] = pd.to_datetime(wd_data['TIMESTAMP'])
            
            # Variable selection for WD
            for group_name, variables in variable_groups['WD'].items():
                # Check which variables are actually available in the data
                available_vars = [var for var in variables if var in wd_data.columns]
                if available_vars:
                    selected_vars = st.multiselect(
                        f"{group_name}:",
                        options=available_vars,
                        default=available_vars,
                        key=f"wd_{group_name}"
                    )
                    
                    if selected_vars:
                        plot_data = wd_data[['TIMESTAMP'] + selected_vars].set_index('TIMESTAMP')
                        st.line_chart(plot_data, use_container_width=True)
        else:
            st.warning("No WD data available")
    
    # Third column: Detailed Variable Viewer
    with col3:
        st.subheader("Detailed Variable Analysis")
        
        # Create a comprehensive list of all available variables
        all_variables = []
        for data_type, df in data_dict.items():
            # Get numeric columns only (excluding metadata columns)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            # Remove timestamp-related columns and metadata
            exclude_cols = ['TIMESTAMP', 'source_file', 'station', 'data_type', 'file_path']
            available_vars = [col for col in numeric_cols if col not in exclude_cols]
            
            for var in available_vars:
                all_variables.append(f"{data_type}_{var}")
        
        if all_variables:
            # Variable selector
            selected_variable = st.selectbox(
                "Select Variable for Detailed Analysis:",
                options=all_variables,
                format_func=lambda x: f"{x.split('_')[0]} - {x.split('_', 1)[1]}"
            )
            
            if selected_variable:
                data_type, variable = selected_variable.split('_', 1)
                df = data_dict[data_type].copy()
                df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                
                st.write(f"**{data_type} - {variable} Time Series**")
                # Plot the selected variable
                plot_data = df[['TIMESTAMP', variable]].set_index('TIMESTAMP')
                st.line_chart(plot_data, use_container_width=True)
                
                # Show recent data points
                st.write("**Recent Data Points:**")
                recent_data = df[['TIMESTAMP', variable]].tail(10)
                st.dataframe(recent_data, use_container_width=True)
                
                st.write(f"**{variable} Statistics**")
                
                # Basic statistics
                stats = df[variable].describe()
                st.write("**Basic Statistics:**")
                st.dataframe(stats, use_container_width=True)
                
                # Data quality info
                total_values = len(df[variable])
                null_values = df[variable].isnull().sum()
                valid_values = total_values - null_values
                
                st.write("**Data Quality:**")
                st.metric("Total Values", total_values)
                st.metric("Valid Values", valid_values)
                st.metric("Null Values", null_values)
                st.metric("Completeness (%)", f"{(valid_values/total_values*100):.1f}%")
                
                # Value range
                if valid_values > 0:
                    min_val = df[variable].min()
                    max_val = df[variable].max()
                    st.write("**Value Range:**")
                    st.metric("Minimum", f"{min_val:.4f}")
                    st.metric("Maximum", f"{max_val:.4f}")
                    st.metric("Range", f"{max_val - min_val:.4f}")
        else:
            st.warning("No variables available for detailed analysis")

    # Data Quality Section
    st.header("Data Quality Overview")
    
    quality_info = []
    for data_type, df in data_dict.items():
        
        # Basic quality metrics
        total_records = len(df)
        null_counts = df.isnull().sum().sum()
        duplicate_records = df.duplicated().sum()
        
        quality_info.append({
            'Station': selected_station.upper(),
            'Data Type': data_type,
            'Total Records': total_records,
            'Null Values': null_counts,
            'Duplicate Records': duplicate_records,
            'Data Completeness (%)': round((total_records - null_counts) / total_records * 100, 2) if total_records > 0 else 0
        })
    
    quality_df = pd.DataFrame(quality_info)
    st.dataframe(quality_df, use_container_width=True)

    # Raw Data Section (collapsible)
    with st.expander("Raw Data Tables"):
        for data_type, df in data_dict.items():
            st.write(f"**{selected_station.upper()} - {data_type} Data**")
            st.dataframe(df.head(100), use_container_width=True)  # Show first 100 rows
            st.write(f"Total records: {len(df)}")

else:
    st.error("No data files found. Please check the data pipeline and ensure files are being generated.")