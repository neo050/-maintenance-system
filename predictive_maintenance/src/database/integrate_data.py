import pandas as pd
import os
from predictive_maintenance.src.utils.logging_util import get_logger
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import yaml

logger = get_logger(__name__)


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


def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None


def save_to_database(engine, data, table_name):
    if engine is None:
        logger.error("Engine is not initialized")
        return
    try:
        with engine.begin() as connection:
            data.to_sql(table_name, connection, if_exists='replace', index=False)
            logger.info(f"Data saved to database table {table_name} successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error saving data to database: {e}")


def integrate_data(real_data_path, simulated_data_path, output_path, db_config_path):
    real_data = load_data(real_data_path)
    if real_data is None:
        logger.error("Failed to load real data")
        return

    simulated_data = load_data(simulated_data_path)
    if simulated_data is None:
        logger.error("Failed to load simulated data")
        return

    combined_data = pd.concat([real_data, simulated_data], ignore_index=True)

    # Remove rows with missing values
    combined_data.dropna(inplace=True)

    combined_data.to_csv(output_path, index=False)
    logger.info(f"Combined data saved to {output_path}")

    # Load database configuration
    config = load_config(db_config_path)
    if config is not None and 'database' in config and 'url' in config['database']:
        engine = create_engine(config['database']['url'])
        save_to_database(engine, combined_data, 'combined_data')
    else:
        logger.error("Database configuration is missing or invalid")


if __name__ == "__main__":
    real_data_file = '../../../data/processed/processed_data_with_lags.csv'
    simulated_data_file = '../../../data/simulation/simulate_processed_sensor_data_with_lags.csv'
    output_file = '../../../data/processed/combined_data.csv'
    db_config_file = '../../../config/database_config.yaml'

    integrate_data(real_data_file, simulated_data_file, output_file, db_config_file)
