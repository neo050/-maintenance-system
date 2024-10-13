import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, chi2_contingency
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """Load the raw data from the specified file path."""
    logging.info(f"Loading data from file: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def calculate_descriptive_stats(data):
    """Calculate descriptive statistics for the dataset."""
    return data.describe()

def perform_ks_test(real_data, simulated_data, columns):
    results = {}
    for col in columns:
        ks_stat, p_value = ks_2samp(real_data[col].dropna(), simulated_data[col].dropna())
        results[col] = {'ks_stat': ks_stat, 'p_value': p_value}
    return results

def perform_chi_square_test(real_data, simulated_data, columns):
    results = {}
    for col in columns:
        contingency_table = pd.crosstab(real_data[col], simulated_data[col])
        chi2, p, dof, ex = chi2_contingency(contingency_table, correction=False)
        results[col] = {'chi2': chi2, 'p_value': p}
    return results

def validate_data(real_data, simulated_data):
    """Validate the simulated data against the real data."""
    continuous_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    categorical_columns = ['Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    ks_test_results = perform_ks_test(real_data, simulated_data, continuous_columns)
    chi_square_test_results = perform_chi_square_test(real_data, simulated_data, categorical_columns)

    return ks_test_results, chi_square_test_results

def simulate_sensor_data(real_data, n_samples):
    np.random.seed(42)  # Fixed random seed for reproducibility

    # Extract real data probabilities for categorical variables
    osf_prob = real_data['OSF'].value_counts(normalize=True).to_dict()
    machine_failure_prob = real_data['Machine failure'].value_counts(normalize=True).to_dict()
    pwf_prob = real_data['PWF'].value_counts(normalize=True).to_dict()

    # Generate data based on real data statistics
    simulated_data = {
        'UDI': np.arange(1, n_samples + 1),
        'Product ID': [f'P{str(i).zfill(5)}' for i in np.random.randint(1, 100, n_samples)],
        'Type': np.random.choice(real_data['Type'], n_samples),
        'Air temperature [K]': np.random.choice(real_data['Air temperature [K]'], n_samples),
        'Process temperature [K]': np.random.choice(real_data['Process temperature [K]'], n_samples),
        'Rotational speed [rpm]': np.random.choice(real_data['Rotational speed [rpm]'], n_samples),
        'Torque [Nm]': np.random.choice(real_data['Torque [Nm]'], n_samples),
        'Tool wear [min]': np.random.choice(real_data['Tool wear [min]'], n_samples),
        'Machine failure': np.random.choice([0, 1], n_samples, p=[machine_failure_prob[0], machine_failure_prob[1]]),
        'TWF': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'HDF': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'PWF': np.random.choice([0, 1], n_samples, p=[pwf_prob[0], pwf_prob[1]]),
        'OSF': np.random.choice([0, 1], n_samples, p=[osf_prob[0], osf_prob[1]]),
        'RNF': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    }

    df_simulated = pd.DataFrame(simulated_data)

    # Feature engineering
    df_simulated['air_temp_diff'] = df_simulated['Process temperature [K]'] - df_simulated['Air temperature [K]']
    df_simulated['power'] = df_simulated['Torque [Nm]'] * df_simulated['Rotational speed [rpm]']

    # Log transformation
    for feature in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
        df_simulated[f'log_{feature}'] = np.log1p(df_simulated[feature])

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_simulated[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']])
    scaled_df = pd.DataFrame(scaled_features, columns=[f'scaled_{col}' for col in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']])
    df_simulated = pd.concat([df_simulated, scaled_df], axis=1)

    # Add lag features
    lags = 10
    lagged_data = pd.concat([df_simulated.shift(i).add_suffix(f'_lag_{i}') for i in range(1, lags + 1)], axis=1)
    df_simulated = pd.concat([df_simulated, lagged_data], axis=1).dropna()

    return df_simulated

def main():
    real_data_file = '../../data/processed/processed_data_with_lags.csv'
    real_data = load_data(real_data_file)

    desired_ks_p_value = 0.05
    desired_chi_square_p_value = 0.05
    max_iterations = 100
    sample_increment = 1000
    success = False
    iteration = 0
    n_samples = 5000  # Larger initial sample size

    while not success and iteration < max_iterations:
        iteration += 1
        logging.info(f"Starting iteration {iteration}")

        simulated_data = simulate_sensor_data(real_data, n_samples)
        ks_test_results, chi_square_test_results = validate_data(real_data, simulated_data)

        # Check if all p-values are above the desired threshold
        ks_p_values = [result['p_value'] for result in ks_test_results.values()]
        chi_square_p_values = [result['p_value'] for result in chi_square_test_results.values()]

        # Log the results for better debugging
        logging.info(f"Iteration {iteration} KS Test Results: {ks_test_results}")
        logging.info(f"Iteration {iteration} Chi-Square Test Results: {chi_square_test_results}")

        # Ensure all conditions are met before stopping
        if all(p > desired_ks_p_value for p in ks_p_values) and all(p > desired_chi_square_p_value for p in chi_square_p_values):
            success = True
            logging.info("Desired results achieved. Saving simulated data.")
            output_path = '../../data/simulation/simulated_data.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            simulated_data.to_csv(output_path, index=False)
        else:
            logging.info("Desired results not achieved. Continuing to next iteration.")
            n_samples += sample_increment  # Increase the number of samples gradually

    if not success:
        logging.error("Maximum iterations reached without achieving desired results.")

if __name__ == "__main__":
    main()