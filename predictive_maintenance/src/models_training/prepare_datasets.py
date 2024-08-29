import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logging_util import get_logger

logger = get_logger(__name__)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def split_data(data, test_size=0.2):
    logger.info(f"Splitting data into train and test sets with test size {test_size}")
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, val_data

if __name__ == "__main__":
    real_data_file = '../data/processed/processed_data_with_lags.csv'
    simulated_data_file = '../data/simulation/simulated_data.csv'
    combined_data_file = '../data/processed/combined_data.csv'

    real_data = load_data(real_data_file)
    simulated_data = load_data(simulated_data_file)
    combined_data = load_data(combined_data_file)

    real_train, real_val = split_data(real_data)
    simulated_train, simulated_val = split_data(simulated_data)
    combined_train, combined_val = split_data(combined_data)

    real_train.to_csv('../data/processed/real_train.csv', index=False)
    real_val.to_csv('../data/processed/real_val.csv', index=False)
    simulated_train.to_csv('../data/processed/simulated_train.csv', index=False)
    simulated_val.to_csv('../data/processed/simulated_val.csv', index=False)
    combined_train.to_csv('../data/processed/combined_train.csv', index=False)
    combined_val.to_csv('../data/processed/combined_val.csv', index=False)

    logger.info("Dataset preparation completed")
