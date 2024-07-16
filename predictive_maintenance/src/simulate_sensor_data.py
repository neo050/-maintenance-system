import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler

from utils.logging_util import get_logger

logger = get_logger(__name__)

def simulate_sensor_data(n_samples=1000, output_path='../data/simulation/simulated_data.csv'):
    np.random.seed(0)

    simulated_data = {
        'UDI': np.arange(1, n_samples + 1),
        'Product ID': [f'P{str(i).zfill(5)}' for i in np.random.randint(1, 100, n_samples)],
        'Type': np.random.choice(['L', 'M', 'H'], n_samples),
        'Air temperature [K]': np.random.uniform(290, 310, n_samples),
        'Process temperature [K]': np.random.uniform(300, 320, n_samples),
        'Rotational speed [rpm]': np.random.uniform(1500, 3000, n_samples),
        'Torque [Nm]': np.random.uniform(10, 50, n_samples),
        'Tool wear [min]': np.random.uniform(0, 300, n_samples),
        'Machine failure': np.random.choice([0, 1], n_samples),
        'TWF': np.random.choice([0, 1], n_samples),
        'HDF': np.random.choice([0, 1], n_samples),
        'PWF': np.random.choice([0, 1], n_samples),
        'OSF': np.random.choice([0, 1], n_samples),
        'RNF': np.random.choice([0, 1], n_samples)
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
    df_simulated = pd.concat([df_simulated, lagged_data], axis=1)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_simulated.to_csv(output_path, index=False)
    logger.info(f"Simulated data saved to {output_path}")

if __name__ == "__main__":
    simulate_sensor_data()
