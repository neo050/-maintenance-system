import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples to generate (same as original dataset)
num_samples = 10000

# Generate Product ID and Type based on proportions (50% L, 30% M, 20% H)
product_types = np.random.choice(['L', 'M', 'H'], size=num_samples, p=[0.5, 0.3, 0.2])
product_ids = [f'{ptype}{np.random.randint(10000, 99999)}' for ptype in product_types]

# Generate air temperature (random walk, normalized)
air_temp = np.random.normal(loc=300, scale=2, size=num_samples)

# Generate process temperature (air temperature + 10, with additional noise)
process_temp = air_temp + 10 + np.random.normal(loc=0, scale=1, size=num_samples)

# Generate rotational speed based on a power of 2860 W, overlaid with noise
rotational_speed = np.random.normal(loc=1500, scale=100, size=num_samples)

# Generate torque values (normally distributed around 40 Nm with SD = 10 Nm)
torque = np.clip(np.random.normal(loc=40, scale=10, size=num_samples), a_min=0, a_max=None)

# Generate tool wear based on product type (H adds 5 min, M adds 3 min, L adds 2 min)
tool_wear = np.array([2 if p == 'L' else 3 if p == 'M' else 5 for p in product_types])
tool_wear = tool_wear + np.random.randint(0, 240, size=num_samples)

# Initialize machine failure (binary) and failure modes
machine_failure = np.zeros(num_samples, dtype=int)
twf = np.zeros(num_samples, dtype=int)
hdf = np.zeros(num_samples, dtype=int)
pwf = np.zeros(num_samples, dtype=int)
osf = np.zeros(num_samples, dtype=int)
rnf = np.zeros(num_samples, dtype=int)

# Apply failure logic based on the conditions

# Tool Wear Failure (TWF): occurs between 200-240 minutes
twf_indices = np.where((tool_wear >= 200) & (tool_wear <= 240))[0]
twf[twf_indices] = 1
# Randomly assign TWF as failure or not
machine_failure[twf_indices] = np.random.choice([0, 1], size=len(twf_indices), p=[0.43, 0.57])

# Heat Dissipation Failure (HDF): difference between air and process temperature < 8.6 K and rotational speed < 1380 rpm
hdf_indices = np.where((process_temp - air_temp < 8.6) & (rotational_speed < 1380))[0]
hdf[hdf_indices] = 1
machine_failure[hdf_indices] = 1

# Power Failure (PWF): power (torque * rotational speed) out of range [3500, 9000 W]
power = torque * (rotational_speed * 2 * np.pi / 60)  # Power in watts (rad/s * torque)
pwf_indices = np.where((power < 3500) | (power > 9000))[0]
pwf[pwf_indices] = 1
machine_failure[pwf_indices] = 1

# Overstrain Failure (OSF): tool wear * torque exceeds threshold based on product type
osf_threshold = {'L': 11000, 'M': 12000, 'H': 13000}
osf_indices = [i for i, p in enumerate(product_types) if tool_wear[i] * torque[i] > osf_threshold[p]]
osf[osf_indices] = 1
machine_failure[osf_indices] = 1

# Random Failures (RNF): 0.1% chance of failure regardless of conditions
rnf_indices = np.random.choice(num_samples, size=int(num_samples * 0.001), replace=False)
rnf[rnf_indices] = 1
machine_failure[rnf_indices] = 1

# Compile the synthetic dataset into a DataFrame
synthetic_data = pd.DataFrame({
    'UDI': np.arange(1, num_samples + 1),
    'Product ID': product_ids,
    'Type': product_types,
    'Air temperature [K]': air_temp,
    'Process temperature [K]': process_temp,
    'Rotational speed [rpm]': rotational_speed,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear,
    'Machine failure': machine_failure,
    'TWF': twf,
    'HDF': hdf,
    'PWF': pwf,
    'OSF': osf,
    'RNF': rnf
})

synthetic_data.to_csv("../../data/simulation/synthetic_data.csv", index=False)