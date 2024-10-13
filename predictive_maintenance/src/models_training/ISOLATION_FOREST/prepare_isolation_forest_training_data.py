import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from sklearn.decomposition import PCA

# Setup logging
logging.basicConfig(filename='../../../logs/data_preparation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Starting data preparation for Isolation Forest Anomaly Detection training')

# Load the data
data_path = '../../../data/processed/processed_data_with_lags.csv'  # Update this for other databases
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

# Select relevant features for anomaly detection
selected_features = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]
# Add the lagged features (if relevant for anomaly detection in time-series data)
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

# Feature scaling (using MinMaxScaler, but StandardScaler can also be used for models like Isolation Forest)
scaler = MinMaxScaler()  # Use StandardScaler() if preferred
data[selected_features] = scaler.fit_transform(data[selected_features])
logging.info('Feature scaling applied using MinMaxScaler.')

# (Optional) Dimensionality Reduction using PCA
pca = PCA(n_components=10)  # Adjust n_components based on your analysis
data_pca = pca.fit_transform(data[selected_features])
logging.info(f'PCA applied to reduce dimensions, resulting in {pca.n_components_} principal components.')

# Replace selected features with PCA output (if applicable)
data_pca_df = pd.DataFrame(data_pca, columns=[f'PCA_{i}' for i in range(pca.n_components_)])
data = pd.concat([data_pca_df, data], axis=1)
logging.info('PCA-transformed features added to the dataset.')

# Split the data into features and target
# The target here is 'Machine failure', which will help evaluate anomaly detection models
X = data[selected_features]  # Features for anomaly detection
y = data['Machine failure']  # Target for evaluation

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f'Data split into training and test sets: '
             f'Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')

# Save the prepared datasets
X_train.to_csv('../../../data/prepare_anomaly_training_data/processed_X_train.csv', index=False)
X_test.to_csv('../../../data/prepare_anomaly_training_data/processed_X_test.csv', index=False)
y_train.to_csv('../../../data/prepare_anomaly_training_data/processed_y_train.csv', index=False)
y_test.to_csv('../../../data/prepare_anomaly_training_data/processed_y_test.csv', index=False)

logging.info('Prepared datasets saved to processed directory.')
logging.info('Data preparation for Isolation Forest Anomaly Detection training completed successfully.')
