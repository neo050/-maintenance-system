import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import save_model
import joblib

# Load the processed data
data = pd.read_csv('../data/processed/processed_data_with_lags.csv')

# Set the number of timesteps for LSTM
timesteps = 10

# Create sequences for LSTM
def create_sequences(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data.iloc[i:(i + timesteps)].values)
        y.append(data.iloc[i + timesteps]['Machine failure'])
    return np.array(X), np.array(y)

# Drop non-numeric columns for LSTM input
data = data.drop(columns=['UDI', 'Product ID', 'Type'])

X, y = create_sequences(data, timesteps)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("Generated data arrays X or y are empty. Check the preprocessing steps.")

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the LSTM model
model.fit(X, y, epochs=50, batch_size=64, verbose=1)

# Save the trained LSTM model
save_model(model, '../models/lstm_model.keras')

# Train Isolation Forest for anomaly detection
isolation_forest = IsolationForest(contamination=0.01)
isolation_forest.fit(data.values)

# Save the trained Isolation Forest model
joblib.dump(isolation_forest, '../models/isolation_forest_model.pkl')
