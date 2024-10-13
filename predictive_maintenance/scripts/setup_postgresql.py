from sqlalchemy import create_engine, Column, Integer, String, Float, MetaData, Table
import pandas as pd
import yaml
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def create_database(config):
    if config is None or 'database' not in config or 'url' not in config['database']:
        logger.error("Database configuration is missing or invalid")
        return None
    try:
        engine = create_engine(config['database']['url'])
        metadata = MetaData()

        processed_data_table = Table(
            'processed_data', metadata,
            Column('id', Integer, primary_key=True),
            Column('UDI', Integer),
            Column('Product_ID', String(50)),
            Column('Type', String(1)),
            Column('Air_temperature_K', Float),
            Column('Process_temperature_K', Float),
            Column('Rotational_speed_rpm', Float),
            Column('Torque_Nm', Float),
            Column('Tool_wear_min', Float),
            Column('Machine_failure', Integer),
            Column('TWF', Integer),
            Column('HDF', Integer),
            Column('PWF', Integer),
            Column('OSF', Integer),
            Column('RNF', Integer),
            Column('air_temp_diff', Float),
            Column('power', Float),
            # Add other columns as needed
        )

        metadata.create_all(engine)
        logger.info("Database and table created successfully")
        return engine
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return None

def save_to_database(engine, data):
    if engine is None:
        logger.error("Engine is not initialized")
        return
    try:
        data.to_sql('processed_data', engine, if_exists='replace', index=False)
        logger.info("Data saved to database successfully")
    except Exception as e:
        logger.error(f"Error saving data to database: {e}")

if __name__ == "__main__":
    config_file = '../config/database_config.yaml'
    config = load_config(config_file)
    engine = create_database(config)

    if engine is not None:
        processed_data_file = '../data/processed/processed_data_with_lags.csv'
        processed_data = pd.read_csv(processed_data_file)
        save_to_database(engine, processed_data)
