import os
import sys
import logging
import json

import yaml
from kafka import KafkaProducer
import pandas as pd
import time

# Remove existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all logs

# Create handlers
log_file_path = '../logs/setup_kafka.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# File handler for logging to a file (DEBUG level and above)
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Stream handler for logging to the console (INFO level and above)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)



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
def create_kafka_producer(kafka_config):
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info("Kafka producer connected.")
        return producer
    except Exception as e:
        logger.error(f"Error connecting to Kafka: {e}")
        sys.exit(1)

def send_failure_predictions(producer, topic, predictions):
    for prediction in predictions:
        try:
            producer.send(topic, prediction)
            producer.flush()
            logger.info(f"Sent prediction for AssetId {prediction['AssetId']}")
            time.sleep(1)  # Throttle the sending if needed
        except Exception as e:
            logger.error(f"Error sending prediction: {e}")

def generate_failure_predictions():
    # Placeholder for actual prediction logic
    # For demonstration, we'll read from the real-time processing results
    predictions_csv_path = '../data/real_time_predictions.csv'
    if os.path.exists(predictions_csv_path):
        df_predictions = pd.read_csv(predictions_csv_path)
        predictions = df_predictions.to_dict('records')
    else:
        logger.warning(f"{predictions_csv_path} not found. Generating dummy data.")
        predictions = [
            {
                'AssetId': 101,
                'PredictedFailureDate': '2023-12-01',
                'RiskScore': 0.85,
                'FailureMode': 'Bearing Wear'
            },
            # Add more predictions as needed
        ]
    return predictions

def main():
    # Load Kafka configuration
    config = load_config('../config/kafka_config.yaml')
    kafka_config = config.get('kafka', {})
    topic = kafka_config.get('failure_predictions_topic', 'failure_predictions')

    producer = create_kafka_producer(kafka_config)

    # Generate or load failure predictions
    predictions = generate_failure_predictions()

    # Send predictions to Kafka
    send_failure_predictions(producer, topic, predictions)

    producer.close()
    logger.info("Kafka producer closed.")

if __name__ == "__main__":
    main()
