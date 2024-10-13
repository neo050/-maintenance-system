import os
import yaml
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Input
from sklearn.ensemble import IsolationForest
from utils.logging_util import get_logger
from io import StringIO

# Initialize logger
logger = get_logger(__name__)

# Load configuration
def load_config(config_file):
    if not os.path.exists(config_file):
        logger.error(f"Configuration file does not exist: {config_file}")
        return None
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_file}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error loading configuration file: {e}")
            return None

config = load_config('../config/model_config.yaml')

# Ensure output directory exists
output_dir = config.get('output_dir', '../models')
os.makedirs(output_dir, exist_ok=True)

def create_lstm_model(input_shape):
    try:
        logger.info("Creating LSTM model")
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(50))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info("LSTM model created successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating LSTM model: {e}")

def create_cnn_model(input_shape):
    try:
        logger.info("Creating CNN model")
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv1D(32, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info("CNN model created successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating CNN model: {e}")

def create_autoencoder_model(input_shape):
    try:
        logger.info("Creating Autoencoder model")
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(input_shape[0], activation='sigmoid'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("Autoencoder model created successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating Autoencoder model: {e}")

def create_isolation_forest_model():
    try:
        logger.info("Creating Isolation Forest model")
        model = IsolationForest(n_estimators=100, contamination=0.1)
        logger.info("Isolation Forest model created successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating Isolation Forest model: {e}")

def log_model_summary(model, model_name):
    try:
        logger.info(f"{model_name} summary:")
        stream = StringIO()
        model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
        summary_str = stream.getvalue()
        summary_str = summary_str.encode('ascii', 'ignore').decode('ascii')  # Simplify to avoid encoding issues
        logger.info("\n" + summary_str)
        stream.close()
    except Exception as e:
        logger.error(f"Error logging summary for {model_name}: {e}")

if __name__ == "__main__":
    logger.info("Starting model creation process")

    lstm_model = create_lstm_model((100, 1))
    cnn_model = create_cnn_model((100, 1))
    autoencoder_model = create_autoencoder_model((100,))
    isolation_forest_model = create_isolation_forest_model()

    logger.info("Model creation process completed")

    if lstm_model:
        log_model_summary(lstm_model, "LSTM model")
    if cnn_model:
        log_model_summary(cnn_model, "CNN model")
    if autoencoder_model:
        log_model_summary(autoencoder_model, "Autoencoder model")

    logger.info("Isolation Forest model created, no summary available for non-keras models")
