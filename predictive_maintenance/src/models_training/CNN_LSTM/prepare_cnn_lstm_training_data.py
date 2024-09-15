import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import logging

# Setup logging
logging.basicConfig(filename='../../../logs/data_preparation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Starting data preparation for CNN-LSTM training')

# Load the data
data_path = '../../../data/processed/combined_data.csv'  # Assuming the combined data is used
data = pd.read_csv(data_path)
logging.info(f'Data loaded from {data_path}, shape: {data.shape}')

# Check for missing values and fill them
if data.isnull().values.any():
    data.fillna(method='ffill', inplace=True)
    logging.info('Missing values detected and filled using forward fill.')
else:
    logging.info('No missing values detected.')

# Verify data types
logging.info(f'Data types:\n{data.dtypes}')

# Select relevant feature columns for CNN-LSTM
selected_features = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]
# Add the lagged features
lagged_features = [f'{col}_lag_{i}' for col in selected_features for i in range(1, 6)]
selected_features += lagged_features
logging.info(f'Selected feature columns: {selected_features}')

# Ensure selected features exist in the data
missing_features = [col for col in selected_features if col not in data.columns]
if missing_features:
    logging.error(f'Missing features in the dataset: {missing_features}')
    raise KeyError(f'Missing features in the dataset: {missing_features}')
else:
    logging.info('All selected features are present in the dataset.')

# Target column
target_column = 'Machine failure'
logging.info(f'Target column: {target_column}')

# Feature scaling
scaler = StandardScaler()
data[selected_features] = scaler.fit_transform(data[selected_features])
logging.info('Feature scaling applied using StandardScaler.')

# Split the data into features and target
X = data[selected_features]
y = data[target_column]

# Reshaping data for CNN-LSTM
# CNN expects 3D input: (samples, timesteps, features), LSTM can also handle 3D input
# Here timesteps = 1 for a simple case, but this can be modified based on your data's temporal nature
X = np.array(X).reshape(X.shape[0], 1, X.shape[1])  # Reshape for CNN-LSTM
logging.info(f'Data reshaped for CNN-LSTM: {X.shape}')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f'Data split into training and test sets: '
             f'Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')

# Save the prepared datasets
np.save('../../../data/prepare_cnn_lstm_training_data/combined_X_train.npy', X_train)
np.save('../../../data/prepare_cnn_lstm_training_data/combined_X_test.npy', X_test)
np.save('../../../data/prepare_cnn_lstm_training_data/combined_y_train.npy', y_train)
np.save('../../../data/prepare_cnn_lstm_training_data/combined_y_test.npy', y_test)

logging.info('Prepared datasets saved to prepare_cnn_lstm_training_data directory.')
logging.info('Data preparation for CNN-LSTM training completed successfully.')
