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

        # Define table schemas
        real_data_table = Table(
            'real_data', metadata,
            Column('UDI', Integer, primary_key=True),
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
        )

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

            # Adding all lag and scaled columns
            Column('log_Air_temperature_K', Float),
            Column('log_Process_temperature_K', Float),
            Column('log_Rotational_speed_rpm', Float),
            Column('log_Torque_Nm', Float),
            Column('log_Tool_wear_min', Float),

            Column('scaled_Air_temperature_K', Float),
            Column('scaled_Process_temperature_K', Float),
            Column('scaled_Rotational_speed_rpm', Float),
            Column('scaled_Torque_Nm', Float),
            Column('scaled_Tool_wear_min', Float),

            Column('lag_1', Float),
            Column('lag_2', Float),
            Column('lag_3', Float),
            Column('lag_4', Float),
            Column('lag_5', Float),

            Column('scaled_Air_temperature_K_lag_1', Float),
            Column('scaled_Process_temperature_K_lag_1', Float),
            Column('scaled_Rotational_speed_rpm_lag_1', Float),
            Column('scaled_Torque_Nm_lag_1', Float),
            Column('scaled_Tool_wear_min_lag_1', Float),

            Column('scaled_Air_temperature_K_lag_2', Float),
            Column('scaled_Process_temperature_K_lag_2', Float),
            Column('scaled_Rotational_speed_rpm_lag_2', Float),
            Column('scaled_Torque_Nm_lag_2', Float),
            Column('scaled_Tool_wear_min_lag_2', Float),

            Column('scaled_Air_temperature_K_lag_3', Float),
            Column('scaled_Process_temperature_K_lag_3', Float),
            Column('scaled_Rotational_speed_rpm_lag_3', Float),
            Column('scaled_Torque_Nm_lag_3', Float),
            Column('scaled_Tool_wear_min_lag_3', Float),

            Column('scaled_Air_temperature_K_lag_4', Float),
            Column('scaled_Process_temperature_K_lag_4', Float),
            Column('scaled_Rotational_speed_rpm_lag_4', Float),
            Column('scaled_Torque_Nm_lag_4', Float),
            Column('scaled_Tool_wear_min_lag_4', Float),

            Column('scaled_Air_temperature_K_lag_5', Float),
            Column('scaled_Process_temperature_K_lag_5', Float),
            Column('scaled_Rotational_speed_rpm_lag_5', Float),
            Column('scaled_Torque_Nm_lag_5', Float),
            Column('scaled_Tool_wear_min_lag_5', Float),

            Column('lag_1_lag_5', Float),
            Column('lag_2_lag_5', Float),
            Column('lag_3_lag_5', Float),
            Column('lag_4_lag_5', Float),
            Column('lag_5_lag_5', Float)
        )

        simulated_data_table = Table(
            'simulated_data', metadata,
            Column('UDI', Integer, primary_key=True),
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

            # Log and scaled columns
            Column('log_Air_temperature_K', Float),
            Column('log_Process_temperature_K', Float),
            Column('log_Rotational_speed_rpm', Float),
            Column('log_Torque_Nm', Float),
            Column('log_Tool_wear_min', Float),

            Column('scaled_Air_temperature_K', Float),
            Column('scaled_Process_temperature_K', Float),
            Column('scaled_Rotational_speed_rpm', Float),
            Column('scaled_Torque_Nm', Float),
            Column('scaled_Tool_wear_min', Float),

            # Lag columns
            Column('lag_1', Float),
            Column('lag_2', Float),
            Column('lag_3', Float),
            Column('lag_4', Float),
            Column('lag_5', Float),

            # Scaled lag columns
            Column('scaled_Air_temperature_K_lag_1', Float),
            Column('scaled_Process_temperature_K_lag_1', Float),
            Column('scaled_Rotational_speed_rpm_lag_1', Float),
            Column('scaled_Torque_Nm_lag_1', Float),
            Column('scaled_Tool_wear_min_lag_1', Float),

            Column('scaled_Air_temperature_K_lag_2', Float),
            Column('scaled_Process_temperature_K_lag_2', Float),
            Column('scaled_Rotational_speed_rpm_lag_2', Float),
            Column('scaled_Torque_Nm_lag_2', Float),
            Column('scaled_Tool_wear_min_lag_2', Float),

            Column('scaled_Air_temperature_K_lag_3', Float),
            Column('scaled_Process_temperature_K_lag_3', Float),
            Column('scaled_Rotational_speed_rpm_lag_3', Float),
            Column('scaled_Torque_Nm_lag_3', Float),
            Column('scaled_Tool_wear_min_lag_3', Float),

            Column('scaled_Air_temperature_K_lag_4', Float),
            Column('scaled_Process_temperature_K_lag_4', Float),
            Column('scaled_Rotational_speed_rpm_lag_4', Float),
            Column('scaled_Torque_Nm_lag_4', Float),
            Column('scaled_Tool_wear_min_lag_4', Float),

            Column('scaled_Air_temperature_K_lag_5', Float),
            Column('scaled_Process_temperature_K_lag_5', Float),
            Column('scaled_Rotational_speed_rpm_lag_5', Float),
            Column('scaled_Torque_Nm_lag_5', Float),
            Column('scaled_Tool_wear_min_lag_5', Float),

            # Further lag columns
            Column('lag_1_lag_5', Float),
            Column('lag_2_lag_5', Float),
            Column('lag_3_lag_5', Float),
            Column('lag_4_lag_5', Float),
            Column('lag_5_lag_5', Float)
        )

        combined_data_table = Table(
            'combined_data', metadata,
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

            # Log and scaled columns
            Column('log_Air_temperature_K', Float),
            Column('log_Process_temperature_K', Float),
            Column('log_Rotational_speed_rpm', Float),
            Column('log_Torque_Nm', Float),
            Column('log_Tool_wear_min', Float),

            Column('scaled_Air_temperature_K', Float),
            Column('scaled_Process_temperature_K', Float),
            Column('scaled_Rotational_speed_rpm', Float),
            Column('scaled_Torque_Nm', Float),
            Column('scaled_Tool_wear_min', Float),

            # Lag columns
            Column('lag_1', Float),
            Column('lag_2', Float),
            Column('lag_3', Float),
            Column('lag_4', Float),
            Column('lag_5', Float),

            # Scaled lag columns
            Column('scaled_Air_temperature_K_lag_1', Float),
            Column('scaled_Process_temperature_K_lag_1', Float),
            Column('scaled_Rotational_speed_rpm_lag_1', Float),
            Column('scaled_Torque_Nm_lag_1', Float),
            Column('scaled_Tool_wear_min_lag_1', Float),

            Column('scaled_Air_temperature_K_lag_2', Float),
            Column('scaled_Process_temperature_K_lag_2', Float),
            Column('scaled_Rotational_speed_rpm_lag_2', Float),
            Column('scaled_Torque_Nm_lag_2', Float),
            Column('scaled_Tool_wear_min_lag_2', Float),

            Column('scaled_Air_temperature_K_lag_3', Float),
            Column('scaled_Process_temperature_K_lag_3', Float),
            Column('scaled_Rotational_speed_rpm_lag_3', Float),
            Column('scaled_Torque_Nm_lag_3', Float),
            Column('scaled_Tool_wear_min_lag_3', Float),

            Column('scaled_Air_temperature_K_lag_4', Float),
            Column('scaled_Process_temperature_K_lag_4', Float),
            Column('scaled_Rotational_speed_rpm_lag_4', Float),
            Column('scaled_Torque_Nm_lag_4', Float),
            Column('scaled_Tool_wear_min_lag_4', Float),

            Column('scaled_Air_temperature_K_lag_5', Float),
            Column('scaled_Process_temperature_K_lag_5', Float),
            Column('scaled_Rotational_speed_rpm_lag_5', Float),
            Column('scaled_Torque_Nm_lag_5', Float),
            Column('scaled_Tool_wear_min_lag_5', Float),

            # Further lag columns
            Column('lag_1_lag_5', Float),
            Column('lag_2_lag_5', Float),
            Column('lag_3_lag_5', Float),
            Column('lag_4_lag_5', Float),
            Column('lag_5_lag_5', Float)
        )

        metadata.create_all(engine)
        logger.info("Database and tables created successfully")
        return engine
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return None

def save_to_database(engine, data, table_name):
    if engine is None:
        logger.error("Engine is not initialized")
        return
    try:
        data.to_sql(table_name, engine, if_exists='replace', index=False)
        logger.info(f"Data saved to database table {table_name} successfully")
    except Exception as e:
        logger.error(f"Error saving data to database: {e}")

if __name__ == "__main__":
    config_file = '../../config/database_config.yaml'
    config = load_config(config_file)
    engine = create_database(config)

    if engine is not None:
        # Load and save real data
        real_data_file = '../../data/raw/ai4i2020.csv'
        real_data = pd.read_csv(real_data_file)
        save_to_database(engine, real_data, 'real_data')

        # Load and save simulated data
        simulated_data_file = '../../data/simulation/synthetic_data.csv'
        simulated_data = pd.read_csv(simulated_data_file)
        save_to_database(engine, simulated_data, 'simulated_data')

        # Load and save combined data
        combined_data_file = '../../data/combined/combined_data.csv'
        combined_data = pd.read_csv(combined_data_file)
        save_to_database(engine, combined_data, 'combined_data')
