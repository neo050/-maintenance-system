import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import gaussian_kde
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_data(original_data, scaler, feature_columns, num_samples):
    """
    Generate synthetic data based on the original data using Kernel Density Estimation (KDE).

    Parameters:
    - original_data (pd.DataFrame): The original dataset.
    - scaler (object): Fitted scaler object used to scale the data.
    - feature_columns (list): List of feature column names.
    - num_samples (int): Number of synthetic samples to generate.

    Returns:
    - synthetic_df (pd.DataFrame): DataFrame containing the synthetic data.
    """
    logger.info("Starting synthetic data generation")

    # Scale the original data
    scaled_data = scaler.transform(original_data[feature_columns])
    logger.info("Data scaled successfully")

    # Generate synthetic data using KDE
    synthetic_data = gaussian_kde(scaled_data.T).resample(num_samples).T
    synthetic_data = scaler.inverse_transform(synthetic_data)
    logger.info("Synthetic data generated using KDE")

    # Clip synthetic data to the range of original data
    for i, feature in enumerate(feature_columns):
        synthetic_data[:, i] = np.clip(synthetic_data[:, i], original_data[feature].min(), original_data[feature].max())
    logger.info("Synthetic data clipped to the range of original data")

    # Create DataFrame from synthetic data
    synthetic_df = pd.DataFrame(synthetic_data, columns=feature_columns)

    # Add non-feature columns by sampling from original data
    for col in original_data.columns:
        if col not in feature_columns:
            synthetic_df[col] = original_data[col].sample(n=num_samples, replace=True).values
    logger.info("Non-feature columns added to synthetic data")

    return synthetic_df


def save_statistics_to_file(original_data, synthetic_data, feature_columns, file_path):
    """
    Save the statistics of the original and synthetic data to a file.

    Parameters:
    - original_data (pd.DataFrame): The original dataset.
    - synthetic_data (pd.DataFrame): The synthetic dataset.
    - feature_columns (list): List of feature column names.
    - file_path (str): Path to the file where the statistics will be saved.
    """
    logger.info("Saving statistics to file")
    try:
        with open(file_path, 'w') as f:
            f.write("Original Data Statistics:\n")
            f.write(original_data[feature_columns].describe().to_string())
            f.write("\n\nSynthetic Data Statistics:\n")
            f.write(synthetic_data[feature_columns].describe().to_string())
        logger.info("Statistics saved successfully to %s", file_path)
    except Exception as e:
        logger.error("Error saving statistics to file: %s", e)


if __name__ == "__main__":
    # Paths to input and output files
    processed_data_path = '../../data/processed/processed_data_with_lags.csv'
    scaler_path = '../../data/processed/scaler.pkl'
    output_file_path = '../../data/datatests/statistics.txt'

    logger.info("Loading original processed data from %s", processed_data_path)
    original_data = pd.read_csv(processed_data_path)
    logger.info("Original data loaded successfully with shape %s", original_data.shape)

    logger.info("Loading scaler from %s", scaler_path)
    scaler = joblib.load(scaler_path)
    logger.info("Scaler loaded successfully")

    # Define the feature columns
    feature_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                       'Tool wear [min]', 'air_temp_diff', 'power']

    # Generate synthetic data
    num_samples = 10000  # Number of synthetic samples to generate
    synthetic_data = generate_synthetic_data(original_data, scaler, feature_columns, num_samples)
    logger.info("Synthetic data generated with shape %s", synthetic_data.shape)

    # Save statistics to file
    save_statistics_to_file(original_data, synthetic_data, feature_columns, output_file_path)
    logger.info("Process finished successfully")
