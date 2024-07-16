import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Ensure the logs directory exists in the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'data_review.log')),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)

logger = get_logger(__name__)

def load_data(file_path):
    """Load the processed data from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def main():
    input_file = os.path.join(BASE_DIR, 'predictive_maintenance/data/processed/processed_data_with_lags.csv')

    # Load the processed data
    processed_data = load_data(input_file)

    # Display the first few rows of the processed data
    print("First few rows of processed data:")
    print(processed_data.head())

    # Check for missing values
    missing_values = processed_data.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values)

    # Verify the new features
    print("\nSummary statistics of new features:")
    new_features = ['air_temp_diff', 'power']
    if all(feature in processed_data.columns for feature in new_features):
        print(processed_data[new_features].describe())
    else:
        logger.error(f"One or more new features are missing: {new_features}")

    # Plot histograms for log-transformed features
    log_features = ['log_Air temperature [K]', 'log_Process temperature [K]', 'log_Rotational speed [rpm]', 'log_Torque [Nm]', 'log_Tool wear [min]']
    for feature in log_features:
        if feature in processed_data.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(processed_data[feature], kde=True)
            plt.title(f'Histogram of {feature}')
            plt.show()
        else:
            logger.error(f"Log-transformed feature missing: {feature}")

    # Plot box plots for scaled features to check outlier removal
    scaled_features = ['scaled_Air temperature [K]', 'scaled_Process temperature [K]', 'scaled_Rotational speed [rpm]', 'scaled_Torque [Nm]', 'scaled_Tool wear [min]']
    for feature in scaled_features:
        if feature in processed_data.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(y=processed_data[feature])
            plt.title(f'Box Plot of {feature}')
            plt.show()
        else:
            logger.error(f"Scaled feature missing: {feature}")

    # Check the summary statistics of scaled features
    print("\nSummary statistics of scaled features:")
    if all(feature in processed_data.columns for feature in scaled_features):
        print(processed_data[scaled_features].describe())
    else:
        logger.error(f"One or more scaled features are missing: {scaled_features}")

    # List lag features
    lag_features = [col for col in processed_data.columns if 'lag' in col]

    # Check the summary statistics of lag features
    print("\nSummary statistics of lag features:")
    if lag_features:
        print(processed_data[lag_features].describe())
    else:
        logger.error("No lag features found in the dataset")

if __name__ == "__main__":
    main()
