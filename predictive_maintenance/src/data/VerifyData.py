import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load data from the specified file path."""
    logging.info(f"Loading data from file: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading data: {file_path} - {e}")
        raise

def check_missing_values(df):
    """Check for missing values in the dataframe."""
    total_missing = df.isnull().sum().sum()
    missing_by_column = df.isnull().sum()
    return total_missing, missing_by_column

def check_data_types(df):
    """Check data types of the dataframe columns."""
    return df.dtypes

def check_value_ranges(df):
    """Check value ranges for numeric columns in the dataframe."""
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    value_ranges = {col: (df[col].min(), df[col].max()) for col in numeric_cols}
    return value_ranges

def check_inconsistent_records(df):
    """Check for inconsistent temperature records and negative torque values."""
    inconsistent_temp_records = df[df['Process temperature [K]'] < df['Air temperature [K]']].shape[0]
    negative_torque_records = df[df['Torque [Nm]'] < 0].shape[0]
    return inconsistent_temp_records, negative_torque_records

def check_duplicate_rows(df):
    """Check for duplicate rows in the dataframe."""
    duplicate_rows = df.duplicated().sum()
    return duplicate_rows

def check_required_columns(df, required_columns):
    """Check if all required columns are present in the dataframe."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns

def main():
    processed_data_path = '../../data/processed/processed_data_with_lags.csv'
    simulated_data_path = '../../data/simulation/simulate_processed_sensor_data_with_lags.csv'

    # Load data
    processed_data = load_data(processed_data_path)
    simulated_data = load_data(simulated_data_path)

    # Define required columns
    required_columns = ['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                        'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
                        'air_temp_diff', 'power', 'log_Air temperature [K]', 'log_Process temperature [K]',
                        'log_Rotational speed [rpm]', 'log_Torque [Nm]', 'log_Tool wear [min]', 'scaled_Air temperature [K]',
                        'scaled_Process temperature [K]', 'scaled_Rotational speed [rpm]', 'scaled_Torque [Nm]',
                        'scaled_Tool wear [min]'] + [f'lag_{i}' for i in range(1, 6)] + [f'UDI_lag_{i}' for i in range(1, 6)] + \
                        [f'Air temperature [K]_lag_{i}' for i in range(1, 6)] + [f'Process temperature [K]_lag_{i}' for i in range(1, 6)] + \
                        [f'Rotational speed [rpm]_lag_{i}' for i in range(1, 6)] + [f'Torque [Nm]_lag_{i}' for i in range(1, 6)] + \
                        [f'Tool wear [min]_lag_{i}' for i in range(1, 6)] + [f'Machine failure_lag_{i}' for i in range(1, 6)] + \
                        [f'TWF_lag_{i}' for i in range(1, 6)] + [f'HDF_lag_{i}' for i in range(1, 6)] + [f'PWF_lag_{i}' for i in range(1, 6)] + \
                        [f'OSF_lag_{i}' for i in range(1, 6)] + [f'RNF_lag_{i}' for i in range(1, 6)] + [f'air_temp_diff_lag_{i}' for i in range(1, 6)] + \
                        [f'power_lag_{i}' for i in range(1, 6)] + [f'log_Air temperature [K]_lag_{i}' for i in range(1, 6)] + \
                        [f'log_Process temperature [K]_lag_{i}' for i in range(1, 6)] + [f'log_Rotational speed [rpm]_lag_{i}' for i in range(1, 6)] + \
                        [f'log_Torque [Nm]_lag_{i}' for i in range(1, 6)] + [f'log_Tool wear [min]_lag_{i}' for i in range(1, 6)] + \
                        [f'scaled_Air temperature [K]_lag_{i}' for i in range(1, 6)] + [f'scaled_Process temperature [K]_lag_{i}' for i in range(1, 6)] + \
                        [f'scaled_Rotational speed [rpm]_lag_{i}' for i in range(1, 6)] + [f'scaled_Torque [Nm]_lag_{i}' for i in range(1, 6)] + \
                        [f'scaled_Tool wear [min]_lag_{i}' for i in range(1, 6)]

    # Processed Data Checks
    logging.info("Starting checks on processed data.")
    total_missing, missing_by_column = check_missing_values(processed_data)
    logging.info(f"Total missing values: {total_missing}")
    logging.info(f"Missing values by column:\n{missing_by_column}")
    logging.info("Data types:")
    logging.info(check_data_types(processed_data))
    logging.info("Value ranges for numeric columns:")
    logging.info(check_value_ranges(processed_data))
    inconsistent_temp_records, negative_torque_records = check_inconsistent_records(processed_data)
    logging.info(f"Number of inconsistent temperature records: {inconsistent_temp_records}")
    logging.info(f"Number of negative torque records: {negative_torque_records}")
    duplicate_rows = check_duplicate_rows(processed_data)
    logging.info(f"Number of duplicate rows: {duplicate_rows}")
    missing_columns = check_required_columns(processed_data, required_columns)
    if missing_columns:
        logging.warning(f"Missing columns: {missing_columns}")
    else:
        logging.info("All required columns are present.")

    # Simulated Data Checks
    logging.info("Starting checks on simulated data.")
    total_missing, missing_by_column = check_missing_values(simulated_data)
    logging.info(f"Total missing values: {total_missing}")
    logging.info(f"Missing values by column:\n{missing_by_column}")
    logging.info("Data types:")
    logging.info(check_data_types(simulated_data))
    logging.info("Value ranges for numeric columns:")
    logging.info(check_value_ranges(simulated_data))
    inconsistent_temp_records, negative_torque_records = check_inconsistent_records(simulated_data)
    logging.info(f"Number of inconsistent temperature records: {inconsistent_temp_records}")
    logging.info(f"Number of negative torque records: {negative_torque_records}")
    duplicate_rows = check_duplicate_rows(simulated_data)
    logging.info(f"Number of duplicate rows: {duplicate_rows}")
    missing_columns = check_required_columns(simulated_data, required_columns)
    if missing_columns:
        logging.warning(f"Missing columns: {missing_columns}")
    else:
        logging.info("All required columns are present.")

if __name__ == "__main__":
    main()
