import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the raw data
data = pd.read_csv('../data/raw/sensor_data.csv')

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Feature engineering
data['air_temp_diff'] = data['Process temperature [K]'] - data['Air temperature [K]']
data['power'] = data['Torque [Nm]'] * data['Rotational speed [rpm]']

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']])
scaled_features_df = pd.DataFrame(scaled_features, columns=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])

# Adding lag features efficiently
lags = 10
lagged_data = pd.concat([scaled_features_df.shift(i).add_suffix(f'_lag_{i}') for i in range(1, lags + 1)], axis=1)

# Concatenate original data with lagged features
processed_data = pd.concat([data, lagged_data], axis=1).dropna()

# Save processed data
processed_data.to_csv('../data/processed/processed_data_with_lags.csv', index=False)
print(f'Processed data shape: {processed_data.shape}')
