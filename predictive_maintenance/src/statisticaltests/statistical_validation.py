import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import StandardScaler
import os
from predictive_maintenance.src.utils.logging_util import get_logger

logger = get_logger(__name__)

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_descriptive_stats(data):
    return data.describe()

def perform_ks_test(real_data, simulated_data, columns):
    results = {}
    for col in columns:
        if col in real_data.columns and col in simulated_data.columns:
            ks_stat, p_value = ks_2samp(real_data[col].dropna(), simulated_data[col].dropna())
            results[col] = {'ks_stat': ks_stat, 'p_value': p_value}
        else:
            logger.warning(f"Column {col} not found in both datasets. Skipping KS test for this column.")
    return results

def perform_chi_square_test(real_data, simulated_data, columns):
    results = {}
    for col in columns:
        if col in real_data.columns and col in simulated_data.columns:
            contingency_table = pd.crosstab(real_data[col], simulated_data[col])
            chi2, p, dof, ex = chi2_contingency(contingency_table, correction=False)
            results[col] = {'chi2': chi2, 'p_value': p}
        else:
            logger.warning(f"Column {col} not found in both datasets. Skipping Chi-Square test for this column.")
    return results

def validate_data(real_data_path, simulated_data_path, output_path):
    real_data = load_data(real_data_path)
    simulated_data = load_data(simulated_data_path)

    # Descriptive Statistics
    real_stats = calculate_descriptive_stats(real_data)
    simulated_stats = calculate_descriptive_stats(simulated_data)

    logger.info("Descriptive Statistics for Real Data:\n" + str(real_stats))
    logger.info("Descriptive Statistics for Simulated Data:\n" + str(simulated_stats))

    # Continuous columns
    continuous_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'air_temp_diff', 'power']

    # Perform KS test
    ks_test_results = perform_ks_test(real_data, simulated_data, continuous_columns)
    logger.info("Kolmogorov-Smirnov Test Results:\n" + str(ks_test_results))

    # Categorical columns
    categorical_columns = ['Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    # Perform Chi-Square test
    chi_square_test_results = perform_chi_square_test(real_data, simulated_data, categorical_columns)
    logger.info("Chi-Square Test Results:\n" + str(chi_square_test_results))

    # Save results to output path
    with open(output_path, 'w') as f:
        f.write("Descriptive Statistics for Real Data:\n" + str(real_stats) + "\n\n")
        f.write("Descriptive Statistics for Simulated Data:\n" + str(simulated_stats) + "\n\n")
        f.write("Kolmogorov-Smirnov Test Results:\n" + str(ks_test_results) + "\n\n")
        f.write("Chi-Square Test Results:\n" + str(chi_square_test_results) + "\n\n")

if __name__ == "__main__":
    real_data_file = '../../data/processed/processed_data_with_lags.csv'
    simulated_data_file = '../../data/simulation/simulate_processed_sensor_data_with_lags.csv'
    output_file = '../../data/validation/statistical_validation_results_updated.txt'

    validate_data(real_data_file, simulated_data_file, output_file)
