import os
import logging
import datetime
import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv

# Initialize Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and Configuration
load_dotenv()

WEATHER_STATIONS = {
    "A-Basin SA-Summit": {
        "id": "CAABT",
        "cols": ["swin", "temp", "dewp", "rh", "wspd", "wdir", "gust", "mxtemp24h", "mntemp24h"]
    },
    "A-Basin SA-Base": {
        "id": "CAABM",
        "cols": ["pcpac", "depth", "snow24h"]
    }
}

ROOT_PATH = Path(__file__).parents[2]
DATA_DIR = ROOT_PATH / "data"

def _fetch_table_data(engine: sqlalchemy.engine.Engine, tables: List[str], start_time: str, st_id: str) -> List[pd.DataFrame]:
    """Internal helper to iterate through tables and fetch data."""
    data_list = []
    for table in tables:
        try:
            query = f"SELECT * FROM {table} WHERE time > '{start_time}' AND staname = '{st_id}'"
            df = pd.read_sql(query, con=engine)
            if not df.empty:
                data_list.append(df.set_index('time'))
            else:
                logger.warning(f"No data found in table {table} for station {st_id}")
        except Exception as e:
            logger.error(f"Error reading table {table}: {e}")
    return data_list

def get_weather_from_sql_db(st_id: str = "CAABT",
                            start_time: Optional[datetime.datetime] = None) -> pd.DataFrame:
    """
    Fetches weather observation data from multiple SQL tables for a specific station.

    Attempts to connect to a local database first. If the local connection fails (e.g., 
    not on the local network), it attempts to establish an SSH tunnel to the database server.

    Args:
        st_id (str): The station identifier in the SQL database. Defaults to "CAABT".
        start_time (Optional[datetime.datetime]): The beginning of the time range. 
            Defaults to 168 hours (7 days) before the current time.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing data from all relevant tables, 
            indexed by time. Returns an empty DataFrame if no data is found.
    """
    db_user = os.getenv('db_user')
    db_password = os.getenv('db_password')
    db_name = os.getenv('db_name')
    local_host = os.getenv('local_host')
    db_port = int(os.getenv('db_port', 3306))
    
    tables = ["obsBattery", "obsHydro", "obsSnow", "obsSolar", "obsWX"]

    if start_time is None:
        start_time = datetime.datetime.now() - datetime.timedelta(hours=168)
        
    # Format time for SQL query
    start_time_str = start_time.strftime("%Y-%m-%d %H:00:00")
    weather_data_frames = []

    con_str = f'mysql+pymysql://{db_user}:{db_password}@{local_host}:{db_port}/{db_name}'
    engine = create_engine(con_str)

    try:
        logger.info(f"Attempting local connection for station {st_id}...")
        weather_data_frames = _fetch_table_data(engine, tables, start_time_str, st_id)
        
    except sqlalchemy.exc.OperationalError:
        logger.info("Local connection failed. Attempting SSH Tunnel...")
        
        ssh_host = os.getenv('ssh_server')
        ssh_key = os.getenv('ssh_pw')
        ssh_port = int(os.getenv('ssh_port', 22))

        try:
            with SSHTunnelForwarder(
                (ssh_host, ssh_port),
                ssh_username='ron',
                ssh_password=ssh_key,
                remote_bind_address=(local_host, int(db_port)) # Usually binds to DB host
            ) as server:
                # Update connection string to use the tunnel's local bind port
                tunnel_con_str = f'mysql+pymysql://{db_user}:{db_password}@127.0.0.1:{server.local_bind_port}/{db_name}'
                tunnel_engine = create_engine(tunnel_con_str)
                weather_data_frames = _fetch_table_data(tunnel_engine, tables, start_time_str, st_id)
        except Exception as e:
            logger.critical(f"Failed to connect via SSH Tunnel: {e}")

    if not weather_data_frames:
        logger.warning(f"No data retrieved for station {st_id}")
        return pd.DataFrame()

    # Join dataframes on index (time)
    return pd.concat(weather_data_frames, axis=1)

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch weather data from SQL and save to CSV.")
    parser.add_argument("-t", "--target", default=str(DATA_DIR),
                        help="Directory to save output files")
    parser.add_argument("-s", "--start-time", help="Start time for data retrieval")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    all_stations_data = []
    if args.start_time:
        try:
            start_time = datetime.datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            logger.error("Invalid start time format. Use 'YYYY-MM-DD HH:MM:SS'.")
            exit(1)
    logger.info("Starting weather data extraction process.")

    for name, info in WEATHER_STATIONS.items():
        logger.info(f"Processing station: {name}")
        df = get_weather_from_sql_db(st_id=info["id"], start_time=start_time)
        
        if not df.empty:
            # Select only requested columns that actually exist in the result
            existing_cols = [c for c in info["cols"] if c in df.columns]
            all_stations_data.append(df[existing_cols])
        else:
            logger.warning(f"Skipping {name} due to empty dataset.")

    if all_stations_data:
        try:
            output_path = Path(args.target) / "weather"
            output_path.mkdir(parents=True, exist_ok=True)
            
            final_df = pd.concat(all_stations_data, axis=1)
            csv_file = output_path / "weather_data.csv"
            final_df.to_csv(csv_file)
            
            logger.info(f"Successfully saved data to {csv_file}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
    else:
        logger.error("No data collected from any stations. CSV not created.")