import pandas as pd
from sqlalchemy import create_engine
import yaml
import os


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def verify_table(db_config_path, table_name):
    config = load_config(db_config_path)
    if config is not None and 'database' in config and 'url' in config['database']:
        engine = create_engine(config['database']['url'])
        table_data = pd.read_sql_table(table_name, engine)
        print(f"First few rows of {table_name} from database:")
        print(table_data.head())
        print(f"\n{table_name} shape: {table_data.shape}")

        # Check for missing values
        missing_values = table_data.isnull().sum()
        print(f"\nMissing values in {table_name}:")
        print(missing_values[missing_values > 0])
    else:
        print("Database configuration is missing or invalid")


if __name__ == "__main__":
    db_config_file = '../../config/database_config.yaml'
    tables_to_verify = ['real_data', 'processed_data', 'simulated_data', 'combined_data']

    for table_name in tables_to_verify:
        verify_table(db_config_file, table_name)
