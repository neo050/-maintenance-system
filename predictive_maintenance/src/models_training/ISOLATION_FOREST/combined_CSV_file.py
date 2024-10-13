import os
import pandas as pd

# Define the directory containing your history files
directory = r'C:\Users\neora\Desktop\Final_project\-maintenance-system\predictive_maintenance\models\anomaly_detection_model_combined'

# Initialize an empty dataframe to collect all the history data
all_history = pd.DataFrame()

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith("history_"):  # Filter for history files
        filepath = os.path.join(directory, filename)

        # Extract the model name, fold number, and contamination level from the filename
        parts = filename.split('_')
        model_name = parts[0]  # Model name is the first part before '_'
        fold_number = parts[2]  # Fold number comes after 'fold' which is always the second part
        contamination_level = parts[4]  # Contamination level is after 'contamination' which is the fourth part

        history_data = pd.read_csv(filepath)


        # Append the dataframe to the master dataframe
        all_history = pd.concat([all_history, history_data], ignore_index=True)

# Define the path to save the combined CSV file
combined_csv_path = os.path.join(directory, 'combined_history.csv')
# Save the combined dataframe to a CSV file
all_history.to_csv(combined_csv_path, index=False)

print(f'combined history file saved to: {combined_csv_path}')
