import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
from utils.logging_util import get_logger
import os

# Get a logger
logger = get_logger('data_normalization_analysis')

# Define file paths

RAW_DATA_PATH = '../data/raw/sensor_data.csv'
PROCESSED_DATA_DIR = '../data/processed/'

# Ensure processed data directory exists
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

# Load data function
def load_data(file_path):
    """Load the raw data from the specified file path."""
    logger.info("Loading data from file: %s", file_path)
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise

# Plotting function
def plot_features(data, features):
    """Plot histograms and box plots for each feature."""
    logger.info('Starting to plot histograms and box plots for features')
    try:
        for feature in features:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            sns.histplot(data[feature], kde=True)
            plt.title(f'Histogram of {feature}')

            plt.subplot(1, 2, 2)
            sns.boxplot(y=data[feature])
            plt.title(f'Box Plot of {feature}')

            plt.show()
    except Exception as e:
        logger.error("Error plotting features: %s", e)
        raise
    logger.info('Completed plotting histograms and box plots for features')

# Normality test function
def normality_tests(data, features):
    """Check normality for each feature using Shapiro-Wilk test."""
    logger.info('Starting normality tests for features')
    try:
        for feature in features:
            stat, p = shapiro(data[feature].dropna())
            logger.info(f'Shapiro-Wilk Test for {feature}: Statistics={stat}, p-value={p}')
            if p > 0.05:
                logger.info(f'{feature} looks normally distributed (fail to reject H0)')
            else:
                logger.info(f'{feature} does not look normally distributed (reject H0)')
    except Exception as e:
        logger.error("Error performing normality tests: %s", e)
        raise

# Normalization function
def normalize_data(data, features):
    """Apply different normalization techniques to the data."""
    normalized_data = {}
    try:
        # Standardization (Z-score normalization)
        logger.info('Applying Standardization (Z-score normalization)')
        scaler = StandardScaler()
        scaled_features_standard = scaler.fit_transform(data[features])
        normalized_data['standard'] = pd.DataFrame(scaled_features_standard, columns=features)

        # Min-Max Scaling
        logger.info('Applying Min-Max Scaling')
        scaler = MinMaxScaler()
        scaled_features_minmax = scaler.fit_transform(data[features])
        normalized_data['minmax'] = pd.DataFrame(scaled_features_minmax, columns=features)

        # Robust Scaling
        logger.info('Applying Robust Scaling')
        scaler = RobustScaler()
        scaled_features_robust = scaler.fit_transform(data[features])
        normalized_data['robust'] = pd.DataFrame(scaled_features_robust, columns=features)

        # Log Transformation
        logger.info('Applying Log Transformation')
        log_transformed_features = np.log1p(data[features])
        normalized_data['log'] = pd.DataFrame(log_transformed_features, columns=features)

    except Exception as e:
        logger.error("Error applying normalization: %s", e)
        raise

    return normalized_data

# Save processed data function
def save_data(normalized_data, output_dir):
    """Save the processed data to the specified directory."""
    logger.info('Saving processed data to %s', output_dir)
    try:
        for key, df in normalized_data.items():
            df.to_csv(os.path.join(output_dir, f'scaled_features_{key}.csv'), index=False)
    except Exception as e:
        logger.error("Error saving processed data: %s", e)
        raise
    logger.info('Data normalization and transformation complete. Check the "processed" folder for output files.')

if __name__ == "__main__":
    logger.info('Starting data normalization analysis pipeline')

    # Load the raw data
    data = load_data(RAW_DATA_PATH)

    # List of features to visualize
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    # Plot histograms and box plots for each feature
    plot_features(data, features)

    # Check normality for each feature
    normality_tests(data, features)

    # Apply normalization techniques
    normalized_data = normalize_data(data, features)

    # Save the processed data for comparison
    save_data(normalized_data, PROCESSED_DATA_DIR)
