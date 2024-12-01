import os
import sys
import yaml
import logging
from kafka import KafkaConsumer
from OpenMaintClient import OpenMaintClient
from datetime import datetime
import json
# Remove existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all logs

# Create handlers
log_file_path = '../logs/openmaint_consumer.log'
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
        sys.exit(1)
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_file}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error loading configuration file: {e}")
            sys.exit(1)

def create_kafka_consumer(kafka_config):
    try:
        bootstrap_servers = kafka_config.get('bootstrap_servers', ['127.0.0.1:9092'])
        logger.info(f"Using bootstrap servers: {bootstrap_servers}")
        consumer = KafkaConsumer(
            kafka_config.get('failure_predictions_topic', 'failure_predictions'),
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='openmaint-consumer-group',
            api_version=(3, 7, 0),  # Adjust to match your Kafka broker version
            reconnect_backoff_max_ms=5000,
            reconnect_backoff_ms=1000,
            request_timeout_ms=30000
        )
        logger.info("Kafka consumer connected.")
        return consumer
    except Exception as e:
        logger.error(f"Error connecting to Kafka: {e}")
        sys.exit(1)



def main():
    # Load configurations
    kafka_config = load_config('../config/kafka_config.yaml').get('kafka', {})
    openmaint_config = load_config('../config/openmaint_config.yaml').get('openmaint', {})

    consumer = create_kafka_consumer(kafka_config)

    # openMAINT API details
    API_BASE_URL = openmaint_config.get('api_url')
    USERNAME = openmaint_config.get('username')
    PASSWORD = openmaint_config.get('password')


    client = OpenMaintClient(API_BASE_URL,USERNAME,PASSWORD)


    try:
        for message in consumer:
            priority_id = client.get_lookup_id('COMMON - Priority', '3')  # Adjust as needed
            type_id = client.get_lookup_id('MaintProcess - Type', 'Damage')
            site_id = client.get_site_id('YourSiteCode')  # Replace with actual site code
            requester_id = client.get_employee_id('Sensor tester', class_name='InternalEmployee')
            ci_id = client.get_ci_id('YourAssetCode')  # Replace with actual asset code

            # Check for missing IDs
            if None in [priority_id, type_id, site_id, requester_id, ci_id]:
                logger.error("Error: One or more IDs could not be retrieved.")

            prediction = message.value
            logger.info(f"Received prediction: {prediction}")

            work_order_data = {
                "_activity": {
                    "_mode": "start"
                },
                "ShortDescr": "Anomaly Detected",
                "Priority": priority_id,
                "OpeningDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ClosureDate": None,
                "Notes": "An anomaly was detected in the sensor readings.",
                "ProcessNotes": message.value,
                "Type": type_id,
                "Requester": {
                    "_id": requester_id,
                    "_type": "InternalEmployee"
                },
                "Site": {
                    "_id": site_id,
                    "_type": "Site"
                },
                "CI": {
                    "_id": ci_id,
                    "_type": "Equipment"
                }
            }

            work_order_id = client.create_work_order(work_order_data)
            logger.info(f"Work order created with ID: {work_order_id}")

            work_order_details = client.get_work_order_details(work_order_id)
            logger.info(f"ork Order Details:\n {json.dumps(work_order_details, indent=4)}")


    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    except Exception as e:
        logger.error(f"Error consuming messages: {e}", exc_info=True)
    finally:
        consumer.close()
        client.logout()
        logger.info("Kafka consumer and HTTP session closed.")

if __name__ == "__main__":
    main()
