# IntegrationWithExistingSystems/openmaint_consumer.py

import os
import sys

import requests
import yaml
import logging
from kafka import KafkaConsumer
from IntegrationWithExistingSystems.OpenMaintClient import OpenMaintClient
from datetime import datetime
import json
import threading
from contextlib import contextmanager
import signal

# Remove existing handlers to prevent duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all logs

# Create handlers
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'openmaint_consumer.log')

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
    """
    Loads a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Exits:
        Exits the program if the configuration file does not exist or fails to parse.
    """
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
    """
    Creates and returns a KafkaConsumer based on the provided configuration.

    Args:
        kafka_config (dict): Kafka configuration dictionary.

    Returns:
        KafkaConsumer: Configured Kafka consumer.

    Exits:
        Exits the program if the Kafka consumer fails to initialize.
    """
    try:
        bootstrap_servers = kafka_config.get('bootstrap_servers', ['localhost:9092'])
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
            request_timeout_ms=30000,
            consumer_timeout_ms=5000  # Reduced timeout to 5 seconds
        )
        logger.info("Kafka consumer connected.")
        return consumer
    except Exception as e:
        logger.error(f"Error connecting to Kafka: {e}")
        sys.exit(1)


def process_message(client, message):
    """
    Processes a single Kafka message and interacts with openMAINT to create a work order.

    Args:
        client (OpenMaintClient): Initialized openMAINT client.
        message (KafkaConsumer.record): Kafka message record.
    """
    try:
        # Retrieve necessary IDs
        priority_id = client.get_lookup_id('COMMON - Priority', '3')  # Adjust as needed
        type_id = client.get_lookup_id('MaintProcess - Type', 'Damage')
        site_id = client.get_site_id('YourSiteCode')  # Replace with actual site code
        requester_id = client.get_employee_id('Sensor tester', class_name='InternalEmployee')
        # ci_id = client.get_ci_id('YourAssetCode')  # Replace with actual asset code

        # Check for missing IDs
        if None in [priority_id, type_id, site_id, requester_id]:
            logger.error("Error: One or more IDs could not be retrieved. Skipping this message.")
            return  # Skip processing this message

        prediction = message.value
        logger.info(f"Received prediction: {prediction}")

        # Construct work order data
        work_order_data = {
            "_activity": {
                "_mode": "start"
            },
            "ShortDescr": "Anomaly Detected",
            "Priority": priority_id,
            "OpeningDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ClosureDate": None,
            "Notes": "An anomaly was detected in the sensor readings.",
            "ProcessNotes": json.dumps(message.value),  # Serialized to JSON string
            "Type": type_id,
            "Requester": {
                "_id": requester_id,
                "_type": "InternalEmployee"
            },
            "Site": {
                "_id": site_id,
                "_type": "Site"
            },
            # Uncomment and configure if CI is required
            # "CI": {
            #     "_id": ci_id,
            #     "_type": "Equipment"
            # }
        }

        # Create work order in openMAINT
        work_order_id = client.create_work_order(work_order_data)
        logger.info(f"Work order created with ID: {work_order_id}")

        # Retrieve and log work order details
        work_order_details = client.get_work_order_details(work_order_id)
        logger.info(f"Work Order Details:\n {json.dumps(work_order_details, indent=4)}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while processing message: {e}")

    except Exception as e:
        logger.error(f"Unexpected error while processing message: {e}", exc_info=True)


@contextmanager
def managed_openmaint_client(api_url, username, password):
    """
    Context manager for OpenMaintClient to ensure proper setup and teardown.

    Args:
        api_url (str): The base URL of the openMAINT API.
        username (str): Username for authentication.
        password (str): Password for authentication.

    Yields:
        OpenMaintClient: An authenticated OpenMaintClient instance.
    """
    client = OpenMaintClient(api_url, username, password)
    try:
        yield client
    finally:
        client.logout()
        logger.info("OpenMAINT client session closed via context manager.")


def openmaint_consumer_main(shutdown_event):
    """
    Main function to run the OpenMaint consumer.
    Consumes messages from the 'failure_predictions' Kafka topic and creates work orders in openMAINT.

    Args:
        shutdown_event (threading.Event): Event to signal shutdown.
    """
    # Load configurations
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'kafka_config.yaml')
    openmaint_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'openmaint_config.yaml')

    kafka_config = load_config(config_path).get('kafka', {})
    openmaint_config = load_config(openmaint_config_path).get('openmaint', {})

    consumer = create_kafka_consumer(kafka_config)

    # openMAINT API details
    API_BASE_URL = openmaint_config.get('api_url')
    USERNAME = openmaint_config.get('username')
    PASSWORD = openmaint_config.get('password')

    # Validate openMAINT configuration
    if not all([API_BASE_URL, USERNAME, PASSWORD]):
        logger.error("OpenMAINT configuration incomplete. Please provide API URL, username, and password.")
        consumer.close()
        sys.exit(1)

    try:
        with managed_openmaint_client(API_BASE_URL, USERNAME, PASSWORD) as client:
            while not shutdown_event.is_set():
                message_pack = consumer.poll(timeout_ms=1000)  # Poll every second
                if not message_pack:
                    logger.debug("No new messages received.")
                    continue
                for tp, messages in message_pack.items():
                    for message in messages:
                        logger.debug(f"Processing message: {message.value}")
                        try:
                            process_message(client, message)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    except Exception as e:
        logger.error(f"Error consuming messages: {e}", exc_info=True)
    else:
        logger.info("No more messages. OpenMaint consumer is stopping.")
    finally:
        consumer.close()
        logger.info("Kafka consumer closed.")


def handle_shutdown(signum, frame, shutdown_event):
    """
    Signal handler to gracefully shut down the consumer.

    Args:
        signum (int): Signal number.
        frame (FrameType): Current stack frame.
        shutdown_event (threading.Event): Event to signal shutdown.
    """
    logger.info(f"Received shutdown signal (signal {signum}). Initiating graceful shutdown...")
    shutdown_event.set()


if __name__ == "__main__":
    import threading

    # Initialize the shutdown event
    shutdown_event = threading.Event()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: handle_shutdown(s, f, shutdown_event))
    signal.signal(signal.SIGTERM, lambda s, f: handle_shutdown(s, f, shutdown_event))

    try:
        openmaint_consumer_main(shutdown_event)
    except Exception as e:
        logger.error(f"An error occurred in OpenMaint consumer: {e}", exc_info=True)
    finally:

        try:
            # Log before closing handlers
            logger.info("openmaint_consumer is shutting down.")

            # Close and remove all handlers associated with the logger
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
                logger.debug(f"Closed and removed handler: {handler}")
        except Exception as e:
            print(f"Error while closing logger: {e}")


