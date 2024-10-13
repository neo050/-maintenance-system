import os
import pandas as pd
import numpy as np
import logging
import yaml
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from kafka import KafkaConsumer
import json
from joblib import load
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Setup logging
log_file_path = '../logs/real_time_processing_CNN_LSTM.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])



def normalize_data(df, method='minmax'):
    logging.info("Normalizing data using method: %s", method)
    try:
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid normalization method. Choose 'minmax', 'standard', or 'robust'.")

        normalized_data = scaler.fit_transform(df)
        normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
        return normalized_df
    except Exception as e:
        logging.error("Error normalizing data: %s", e)
        raise


# Initialize global variables
df = pd.DataFrame()
feature_columns = []

# Initialize models dictionary
models = {
    'cnn_lstm': [],
    'standard_scaler': None,
    'standard_scaler_fitted': False
}

# CNN-LSTM model features
trained_features = [
    'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Air temperature [K]_lag_1', 'Air temperature [K]_lag_2', 'Air temperature [K]_lag_3', 'Air temperature [K]_lag_4', 'Air temperature [K]_lag_5',
    'Process temperature [K]_lag_1', 'Process temperature [K]_lag_2', 'Process temperature [K]_lag_3', 'Process temperature [K]_lag_4', 'Process temperature [K]_lag_5',
    'Rotational speed [rpm]_lag_1', 'Rotational speed [rpm]_lag_2', 'Rotational speed [rpm]_lag_3', 'Rotational speed [rpm]_lag_4', 'Rotational speed [rpm]_lag_5',
    'Torque [Nm]_lag_1', 'Torque [Nm]_lag_2', 'Torque [Nm]_lag_3', 'Torque [Nm]_lag_4', 'Torque [Nm]_lag_5',
    'Tool wear [min]_lag_1', 'Tool wear [min]_lag_2', 'Tool wear [min]_lag_3', 'Tool wear [min]_lag_4', 'Tool wear [min]_lag_5'
]
# Keep a buffer of the last N data points
sequence_length = 1  # Adjust based on your training time steps
data_buffer = []
def create_consumer(kafka_config, topics):
    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=kafka_config['bootstrap_servers'],
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    )
    return consumer

# Load CNN-LSTM models
def load_cnn_lstm_models(models_dir):
    cnn_lstm_model_files = [os.path.join(models_dir, file) for file in os.listdir(models_dir) if file.endswith('.keras')]
    models['cnn_lstm'] = [load_model(file) for file in cnn_lstm_model_files]
    logging.info("CNN-LSTM models loaded successfully.")
    logging.info(f"Model input shape: {models['cnn_lstm'][0].input_shape}")

# Prepare classification data for CNN-LSTM
def prepare_cnn_lstm_data(df_sequence, models, normalization_method='minmax'):
    global feature_columns

    feature_columns = [col for col in trained_features if col in df_sequence.columns]
    logging.info(f"Incoming DataFrame shape: {df_sequence.shape}")

    df_features = df_sequence[feature_columns]

    # Convert feature columns to numeric
    df_features = df_features.apply(pd.to_numeric, errors='coerce')

    # Fill NaNs
    df_features.fillna(method='ffill', inplace=True)
    df_features.fillna(method='bfill', inplace=True)
    df_features.fillna(0, inplace=True)

    if df_features.isnull().values.any():
        logging.error("DataFrame contains NaNs after filling. Cannot proceed.")
        return None

    if df_features.empty:
        logging.info("DataFrame is empty after handling NaNs. Waiting for more data.")
        return None

    # Normalize data
    df_features = normalize_data(df_features, method=normalization_method)

    # Use the existing StandardScaler if available
    if models['standard_scaler_fitted']:
        df_features_scaled = models['standard_scaler'].transform(df_features)
    else:
        models['standard_scaler'] = StandardScaler()
        df_features_scaled = models['standard_scaler'].fit_transform(df_features)
        models['standard_scaler_fitted'] = True

    # Reshape to match model input
    X_cnn_lstm = df_features_scaled.reshape(1, sequence_length, -1)
    logging.info(f"Input shape for model: {X_cnn_lstm.shape}")
    return X_cnn_lstm





def process_incoming_data(data):
    global data_buffer
    df_new = pd.DataFrame([data])

    # Keep only 1 data point (as per the model's training)
    data_buffer = [df_new]

    # Concatenate buffer into a DataFrame
    df_sequence = pd.concat(data_buffer, ignore_index=True)

    # Prepare data for CNN-LSTM with the correct sequence length of 1
    X_cnn_lstm = prepare_cnn_lstm_data(df_sequence, models)

    if X_cnn_lstm is None:
        return

    # Get prediction from the model
    cnn_lstm_preds = [model.predict(X_cnn_lstm)[0][0] for model in models['cnn_lstm']]
    logging.info(f"CNN-LSTM model predictions: {cnn_lstm_preds}")
    average_prediction = np.mean(cnn_lstm_preds)

    # Use your predetermined threshold
    optimal_threshold = 0.5
    final_prediction = 1 if average_prediction >= optimal_threshold else 0

    logging.info(f"Prediction scores from individual models: {cnn_lstm_preds}")
    logging.info(f"Average prediction score: {average_prediction}")
    logging.info(f"Final Prediction: {final_prediction}")


def load_kafka_config():
    with open('../config/kafka_config.yaml', 'r') as file:
        kafka_config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from ../config/kafka_config.yaml")
    return kafka_config


if __name__ == "__main__":
    models_dir = '../models/cnn_lstm_model_simulation'
    load_cnn_lstm_models(models_dir)

    try:
        models['standard_scaler'] = load('../models/standard_scaler.joblib')
        models['standard_scaler_fitted'] = True
        logging.info("StandardScaler loaded from file.")
    except Exception as e:
       pass

    kafka_config = load_kafka_config()
    topic = kafka_config["topics"]
    consumer = create_consumer(kafka_config, topic)

    logging.info("Kafka consumer initialized and listening for data...")

    message_limit = 100  # Adjust based on how many messages you expect
    message_count = 0

    try:
        for message in consumer:
            data = message.value
            process_incoming_data(data)
            message_count += 1
            logging.info(f'count message , {message_count}')
            if message_count >= message_limit:
                logging.info("Reached message limit, stopping the consumer.")
                break
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
