import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
data_file_path = '../../data/processed/processed_data_with_lags.csv'
logging.info(f'Loading data from file: {data_file_path}')
data = pd.read_csv(data_file_path)


# Function to generate synthetic data (placeholder)
def generate_synthetic_data(df, feature_columns, scaler):
    logging.info("Generating synthetic data")
    synthetic_data = pd.DataFrame(np.random.randn(*df[feature_columns].shape), columns=feature_columns)
    synthetic_data = scaler.inverse_transform(synthetic_data)
    synthetic_data = pd.DataFrame(synthetic_data, columns=feature_columns)
    return synthetic_data


# Function to preprocess data
def preprocess_data(data):
    logging.info('Starting preprocessing pipeline')

    # Filling missing values
    logging.info('Filling missing values')
    data.fillna(method='ffill', inplace=True)
    logging.info(f'Shape after filling missing values: {data.shape}')

    # Performing feature engineering (placeholder)
    logging.info('Performing feature engineering')
    # Add your feature engineering code here
    logging.info(f'Shape after feature engineering: {data.shape}')

    # Converting to numeric
    logging.info('Shape after converting to numeric: {data.shape}')

    # Applying log transformation
    logging.info('Applying log transformation to handle skewness')
    data = data.apply(np.log1p)
    logging.info(f'Shape after log transformation: {data.shape}')

    # Removing outliers
    logging.info('Removing outliers')
    # Add your outlier removal code here
    logging.info(f'Shape after removing outliers: {data.shape}')

    return data


# Function to scale data
def scale_data(data):
    logging.info('Fitting scaler using method: minmax')
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    logging.info(f'Shape after scaling: {data_scaled.shape}')
    return pd.DataFrame(data_scaled, columns=data.columns)


# Function to add lag features
def add_lag_features(data, lags=5):
    for col in data.columns:
        for i in range(1, lags + 1):
            data[f'{col}_lag_{i}'] = data[col].shift(i)
    return data


# Main loop
max_iterations = 30
for iteration in range(1, max_iterations + 1):
    logging.info(f'Starting iteration {iteration}')

    # Generate synthetic data
    logging.info('Generating synthetic data')
    synthetic_data = generate_synthetic_data(data,['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]','Torque [Nm]', 'Tool wear [min]', 'air_temp_diff', 'power'],joblib.load('../../data/processed/scaler.pkl'))

    # Preprocess data
    processed_data = preprocess_data(synthetic_data)

    # Scale data
    scaled_data = scale_data(processed_data)

    # Add lag features
    lagged_data = add_lag_features(scaled_data)
    logging.info(f'Number of NaNs before dropping: {lagged_data.isna().sum().sum()}')

    # Handle NaNs
    lagged_data.fillna(method='ffill', inplace=True)
    logging.info(f'Shape after forward fill: {lagged_data.shape}')

    lagged_data.dropna(inplace=True)
    logging.info(f'Shape after dropping NaNs: {lagged_data.shape}')

    if lagged_data.empty:
        logging.error(
            f'Iteration {iteration} failed: Data is empty after adding lag features and dropping NaNs. Check the lagging process.')
    else:
        logging.info(f'Iteration {iteration} succeeded')
        break

logging.error('Maximum iterations reached without achieving desired results.')
