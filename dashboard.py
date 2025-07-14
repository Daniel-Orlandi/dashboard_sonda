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

# Function to get the latest file for each station and data type
def get_latest_files():
    """Get the latest parquet file for each station and data type (SD, MD, WD)"""
    interim_dir = os.path.expanduser("data/interim")
    
    # Look for station subdirectories
    station_dirs = [d for d in os.listdir(interim_dir) if os.path.isdir(os.path.join(interim_dir, d))]
    
    latest_files = {}
    
    for station in station_dirs:
        station_path = os.path.join(interim_dir, station)
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
        for data_type, file_list in files_by_type.items():
            if file_list:
                # Sort by modification time and get the latest
                latest_file = max(file_list, key=os.path.getmtime)
                latest_files[f"{station}_{data_type}"] = latest_file
    
    return latest_files

# Function to load data from latest files
def load_latest_data():
    """Load data from the latest files for each station and data type"""
    latest_files = get_latest_files()
    
    if not latest_files:
        st.error("No data files found in interim directory")
        return None
    
    data_dict = {}
    
    for key, file_path in latest_files.items():
        try:
            df = pd.read_parquet(file_path)
            # Add metadata columns
            df['source_file'] = os.path.basename(file_path)
            df['station'] = key.split('_')[0]
            df['data_type'] = key.split('_')[1]
            df['file_path'] = file_path
            data_dict[key] = df
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
    
    return data_dict

# Load data
data_dict = load_latest_data()

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
    for key, df in data_dict.items():
        station, data_type = key.split('_')
        file_info.append({
            'Station': station.upper(),
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
    
    # Create three columns for SD, MD, WD
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Solar Data (SD)")
        if any('SD' in key for key in data_dict.keys()):
            sd_key = [key for key in data_dict.keys() if 'SD' in key][0]
            sd_data = data_dict[sd_key].copy()
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
    
    with col2:
        st.subheader("Meteorological Data (MD)")
        if any('MD' in key for key in data_dict.keys()):
            md_key = [key for key in data_dict.keys() if 'MD' in key][0]
            md_data = data_dict[md_key].copy()
            md_data['TIMESTAMP'] = pd.to_datetime(md_data['TIMESTAMP'])
            
            # Variable selection for MD
            st.write("**Select Variables to Plot:**")
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
    
    with col3:
        st.subheader("Wind Data (WD)")
        if any('WD' in key for key in data_dict.keys()):
            wd_key = [key for key in data_dict.keys() if 'WD' in key][0]
            wd_data = data_dict[wd_key].copy()
            wd_data['TIMESTAMP'] = pd.to_datetime(wd_data['TIMESTAMP'])
            
            # Variable selection for WD
            st.write("**Select Variables to Plot:**")
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

    # Data Quality Section
    st.header("Data Quality Overview")
    
    quality_info = []
    for key, df in data_dict.items():
        station, data_type = key.split('_')
        
        # Basic quality metrics
        total_records = len(df)
        null_counts = df.isnull().sum().sum()
        duplicate_records = df.duplicated().sum()
        
        quality_info.append({
            'Station': station.upper(),
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
        for key, df in data_dict.items():
            station, data_type = key.split('_')
            st.write(f"**{station.upper()} - {data_type} Data**")
            st.dataframe(df.head(100), use_container_width=True)  # Show first 100 rows
            st.write(f"Total records: {len(df)}")

else:
    st.error("No data files found. Please check the data pipeline and ensure files are being generated.")