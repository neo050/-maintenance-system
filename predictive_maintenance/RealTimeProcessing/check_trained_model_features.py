import logging
import os
import numpy as np
from tensorflow.keras.models import load_model

# Setup logging
log_file_path = './check_trained_model_features.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])


# Function to check trained features
def check_trained_features(model_path, feature_columns):
    try:
        # Load the trained model
        logging.info(f"Loading model from {model_path}")
        model = load_model(model_path)

        # Access model input shape
        input_shape = model.input_shape  # Using model.input_shape instead of layer input_shape
        logging.info(f"Model input shape: {input_shape}")

        # First layer's weights shape (corresponds to the number of input features)
        first_layer_weights_shape = model.layers[0].get_weights()[0].shape
        logging.info(f"First layer weights shape: {first_layer_weights_shape}")

        num_features = first_layer_weights_shape[1]  # This is the number of features
        logging.info(f"Number of features from model input shape: {num_features}")

        # Compare with config
        logging.info(f"Number of features defined in config: {len(feature_columns)}")

        if num_features != len(feature_columns):
            logging.error("Mismatch between the number of features in the model and the config.")
            return False

        logging.info(f"Features from config: {feature_columns}")

        # Output exact features used for training
        return feature_columns

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Define the model path
    model_path = '/predictive_maintenance/models/cnn_lstm_model_simulation/simulation_fold_1.keras'
    data_path = '/predictive_maintenance/data/prepare_cnn_lstm_training_data/simulation_X_train.npy'
    # Define the features used for training (from the config)
    feature_columns = [
        'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Air temperature [K]_lag_1', 'Air temperature [K]_lag_2', 'Air temperature [K]_lag_3',
        'Air temperature [K]_lag_4',
        'Air temperature [K]_lag_5', 'Process temperature [K]_lag_1', 'Process temperature [K]_lag_2',
        'Process temperature [K]_lag_3',
        'Process temperature [K]_lag_4', 'Process temperature [K]_lag_5', 'Rotational speed [rpm]_lag_1',
        'Rotational speed [rpm]_lag_2', 'Rotational speed [rpm]_lag_3', 'Rotational speed [rpm]_lag_4',
        'Rotational speed [rpm]_lag_5',
        'Torque [Nm]_lag_1', 'Torque [Nm]_lag_2', 'Torque [Nm]_lag_3', 'Torque [Nm]_lag_4', 'Torque [Nm]_lag_5',
        'Tool wear [min]_lag_1', 'Tool wear [min]_lag_2', 'Tool wear [min]_lag_3', 'Tool wear [min]_lag_4',
        'Tool wear [min]_lag_5'
    ]

    # Call the function
    trained_features = check_trained_features(model_path, feature_columns)

    if trained_features:
        logging.info(f"Trained features: {trained_features}")
    else:
        logging.error("Failed to extract features from the trained model.")

