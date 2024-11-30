# File: maintenance-system/predictive_maintenance/src/integration/setup_postgresql.py
import os
import sys
import psycopg2
import pandas as pd
import logging
from datetime import datetime
import yaml
# Remove any existing handlers
from sqlalchemy import create_engine

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all logs

# Create handlers
log_file_path = '../logs/setup_postgresql.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# File handler for logging to a file (DEBUG level and above)
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Stream handler for logging to the console (INFO level and above)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def load_config(config_file):
    if not os.path.exists(config_file):
        logger.error(f"Configuration file does not exist: {config_file}")
        return None
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_file}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error loading configuration file: {e}")
            return None
def get_db_connection(config):
    if config is None or 'openmaint' not in config or 'url' not in config['openmaint']:
        logger.error("Database configuration is missing or invalid")
        return None
    try:
        engine = create_engine(config['openmaint']['url'])
        connection = engine.connect()
        logger.info("Database connection established.")
        return connection
    except Exception as e:
        logger.error(f"Error creating database engine: {e}")
        return None

def fetch_assets(engine):
    query = """
    SELECT
        asset."Id" AS asset_id,
        asset."Code" AS asset_code,
        asset."Description" AS asset_description,
        asset."Type" AS asset_type,
        asset."InstallationDate" AS installation_date,
        asset."WarrantyEnd" AS warranty_expiration,
        asset."Notes" AS notes
    FROM "Asset" AS asset
    WHERE asset."Status" = 'A'
    """
    try:
        df_assets = pd.read_sql_query(query, engine)
        logger.info(f"Fetched {len(df_assets)} assets from the database.")
        return df_assets
    except Exception as e:
        logger.error(f"Error fetching assets: {e}")
        return pd.DataFrame()
def fetch_maintenance_records(engine):
    query = """
    SELECT
        cm."Id" AS maintenance_id,
        cm."Code" AS maintenance_code,
        cm."Description" AS maintenance_description,
        cm."CI" AS asset_id,
        asset."Code" AS asset_code,
        asset."Description" AS asset_description,
        cm."ProcessType" AS process_type,
        cm."OpeningDate" AS start_date,
        cm."ClosureDate" AS end_date,
        cm."Notes" AS notes,
        cm."Status" AS status
    FROM "CorrectiveMaint" AS cm
    LEFT JOIN "Asset" AS asset ON cm."CI" = asset."Id"
    WHERE cm."Status" = 'A'
    """
    try:
        df_maintenance = pd.read_sql_query(query, engine)
        logger.info(f"Fetched {len(df_maintenance)} maintenance records from the database.")
        return df_maintenance
    except Exception as e:
        logger.error(f"Error fetching maintenance records: {e}")
        return pd.DataFrame()
def fetch_preventive_maintenance_records(engine):
    query = """
    SELECT
        pm."Id" AS maintenance_id,
        pm."Code" AS maintenance_code,
        pm."Description" AS maintenance_description,
        pm."CI" AS asset_id,
        asset."Code" AS asset_code,
        asset."Description" AS asset_description,
        pm."ProcessType" AS process_type,
        pm."OpeningDate" AS start_date,
        pm."ClosureDate" AS end_date,
        pm."Notes" AS notes,
        pm."Status" AS status
    FROM "PreventiveMaint" AS pm
    LEFT JOIN "Asset" AS asset ON pm."CI" = asset."Id"
    WHERE pm."Status" = 'A'
    """
    try:
        df_preventive = pd.read_sql_query(query, engine)
        logger.info(f"Fetched {len(df_preventive)} preventive maintenance records from the database.")
        return df_preventive
    except Exception as e:
        logger.error(f"Error fetching preventive maintenance records: {e}")
        return pd.DataFrame()

def main():
    # Load database configuration
    config = load_config('../config/database_config.yaml')
    engine = get_db_connection(config)

    if engine is None:
        logger.error("Engine is not initialized. Exiting.")
        sys.exit(1)

    # Fetch assets
    df_assets = fetch_assets(engine)
    if not df_assets.empty:
        assets_csv_path = '../data/integration/assets.csv'
        os.makedirs(os.path.dirname(assets_csv_path), exist_ok=True)
        df_assets.to_csv(assets_csv_path, index=False)
        logger.info(f"Assets data saved to {assets_csv_path}")
    else:
        logger.warning("No assets data fetched.")

    # Fetch maintenance records
    df_maintenance = fetch_maintenance_records(engine)
    if not df_maintenance.empty:
        maintenance_csv_path = '../data/integration/maintenance_records.csv'
        os.makedirs(os.path.dirname(maintenance_csv_path), exist_ok=True)
        df_maintenance.to_csv(maintenance_csv_path, index=False)
        logger.info(f"Maintenance records data saved to {maintenance_csv_path}")
    else:
        logger.warning("No maintenance records data fetched.")

    #Optionally, fetch preventive maintenance records
    df_preventive = fetch_preventive_maintenance_records(engine)
    if not df_preventive.empty:
         preventive_csv_path = '../data/integration/preventive_maintenance_records.csv'
         os.makedirs(os.path.dirname(preventive_csv_path), exist_ok=True)
         df_preventive.to_csv(preventive_csv_path, index=False)
         logger.info(f"Preventive maintenance records data saved to {preventive_csv_path}")
    else:
        logger.warning("No preventive maintenance records data fetched.")

    # Dispose of the engine
    engine.close()
    logger.info("Database engine disposed.")

if __name__ == "__main__":
    main()
