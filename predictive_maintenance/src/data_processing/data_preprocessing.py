import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    logging.info("Loading data from file: %s", file_path)
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def fill_missing_values(df):
    logging.info("Filling missing values")
    try:
        return df.interpolate(method='linear', limit_direction='forward')
    except Exception as e:
        logging.error("Error filling missing values: %s", e)
        raise

def feature_engineering(df):
    logging.info("Performing feature engineering")
    try:
        df['air_temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
        df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
        return df
    except Exception as e:
        logging.error("Error in feature engineering: %s", e)
        raise

def log_transform(df, feature_columns):
    logging.info("Applying log transformation to handle skewness")
    try:
        for feature in feature_columns:
            df[f'log_{feature}'] = np.log1p(df[feature])
        return df
    except Exception as e:
        logging.error("Error applying log transformation: %s", e)
        raise

def remove_outliers(df, feature_columns):
    logging.info("Removing outliers")
    try:
        for feature in feature_columns:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))]
        return df
    except Exception as e:
        logging.error("Error removing outliers: %s", e)
        raise

def scale_features(df, feature_columns, method='standard'):
    logging.info("Scaling features using method: %s", method)
    try:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standard', 'minmax', or 'robust'.")

        scaled_features = scaler.fit_transform(df[feature_columns])
        scaled_df = pd.DataFrame(scaled_features, columns=[f'scaled_{col}' for col in feature_columns])
        df = pd.concat([df, scaled_df], axis=1)
        return df
    except Exception as e:
        logging.error("Error scaling features: %s", e)
        raise

def add_lag_features(df, lags):
    logging.info("Adding lag features")
    try:
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['UDI'].shift(i)
        lagged_data = pd.concat([df.shift(i).add_suffix(f'_lag_{i}') for i in range(1, lags + 1)], axis=1)
        df = pd.concat([df, lagged_data], axis=1)
        return df
    except Exception as e:
        logging.error("Error adding lag features: %s", e)
        raise

def normalize_data(df, method='minmax'):
    logging.info("Normalizing data using method: %s", method)
    try:
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid normalization method. Choose 'minmax', 'standard', or 'robust'.")

        normalized_data = scaler.fit_transform(df)
        normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
        return normalized_df
    except Exception as e:
        logging.error("Error normalizing data: %s", e)
        raise

def preprocess_data(file_path, output_path, feature_columns, lags=5):
    logging.info("Starting preprocessing pipeline")

    data = load_data(file_path)
    data = fill_missing_values(data)
    data = feature_engineering(data)
    data = log_transform(data, feature_columns)
    data = remove_outliers(data, feature_columns)
    data = scale_features(data, feature_columns)
    data = add_lag_features(data, lags)

    # Normalize data (additional step if necessary)
    normalized_columns = [col for col in data.columns if col.startswith('scaled_') or col.startswith('log_')]
    data[normalized_columns] = normalize_data(data[normalized_columns])

    try:
        data.dropna().to_csv(output_path, index=False)
        logging.info(f'Processed data saved to: {output_path}')
        logging.info(f'Processed data shape: {data.shape}')
    except Exception as e:
        logging.error("Error saving processed data: %s", e)
        raise

if __name__ == "__main__":
    input_file = '../../data/raw/sensor_data.csv'
    output_file = '../../data/processed/processed_data_with_lags.csv'
    feature_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    preprocess_data(input_file, output_file, feature_columns)
