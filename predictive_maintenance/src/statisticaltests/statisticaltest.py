import pandas as pd
from scipy import stats
import logging

# Load the datasets
original_data_path = '../../data/processed/processed_data_with_lags.csv'
synthetic_data_path = '../../data/processed/processed_synthetic_data_with_lags.csv'

original_data = pd.read_csv(original_data_path)
synthetic_data = pd.read_csv(synthetic_data_path)

# Select feature columns for comparison
feature_columns = [
    'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
    'Torque [Nm]', 'Tool wear [min]', 'air_temp_diff', 'power'
]

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to perform t-test and KS-test
def compare_datasets(original, synthetic, features):
    results = {
        'Feature': [],
        'Original Mean': [],
        'Synthetic Mean': [],
        'Original Std': [],
        'Synthetic Std': [],
        'T-test p-value': [],
        'KS-test p-value': []
    }

    for feature in features:
        orig_mean = original[feature].mean()
        synth_mean = synthetic[feature].mean()
        orig_std = original[feature].std()
        synth_std = synthetic[feature].std()

        t_test_p = stats.ttest_ind(original[feature], synthetic[feature])[1]
        ks_test_p = stats.ks_2samp(original[feature], synthetic[feature])[1]

        results['Feature'].append(feature)
        results['Original Mean'].append(orig_mean)
        results['Synthetic Mean'].append(synth_mean)
        results['Original Std'].append(orig_std)
        results['Synthetic Std'].append(synth_std)
        results['T-test p-value'].append(t_test_p)
        results['KS-test p-value'].append(ks_test_p)

        logger.info(f'Feature: {feature}, T-test p-value: {t_test_p}, KS-test p-value: {ks_test_p}')

    return pd.DataFrame(results)


# Perform the comparison
comparison_results = compare_datasets(original_data, synthetic_data, feature_columns)

# Save the comparison results to a CSV file
comparison_results_path = '../../data/datatests/comparison_results.csv'
comparison_results.to_csv(comparison_results_path, index=False)

logger.info(f'Comparison results saved to {comparison_results_path}')
