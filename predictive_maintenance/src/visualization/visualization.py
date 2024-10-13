import pandas as pd
import matplotlib.pyplot as plt

# Load the processed data
data = pd.read_csv('../data/processed/processed_data_with_lags.csv')

# Plot sensor data
plt.figure(figsize=(10, 6))
plt.plot(data['Air temperature [K]'], label='Air Temperature [K]')
plt.plot(data['Process temperature [K]'], label='Process Temperature [K]')
plt.plot(data['Rotational speed [rpm]'], label='Rotational Speed [rpm]')
plt.plot(data['Torque [Nm]'], label='Torque [Nm]')
plt.plot(data['Tool wear [min]'], label='Tool Wear [min]')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Sensor Data Over Time')
plt.show()
