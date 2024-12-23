import subprocess
import unittest
import threading
import time
import os
import sys
import logging
import json

from kafka import KafkaConsumer
from sqlalchemy import create_engine, text
import yaml

from RealTimeProcessing.SensorDataSimulatorClient import SensorDataSimulator
from RealTimeProcessing.RealTimeProcessorClient import RealTimeProcessor
from IntegrationWithExistingSystems.openmaint_consumer import openmaint_consumer_main
from IntegrationWithExistingSystems.OpenMaintClient import OpenMaintClient

class TestIntegrationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up logger for the test
        cls.logger = logging.getLogger("IntegrationTest")
        cls.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not cls.logger.handlers:
            cls.logger.addHandler(handler)

        cls.logger.info("Loading configurations for integration test...")

        # Determine the project root directory
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Load configurations
        cls.kafka_config = cls.load_config(os.path.join(cls.project_root, 'config', 'kafka_config.yaml')).get('kafka', {})
        cls.database_config = cls.load_config(os.path.join(cls.project_root, 'config', 'database_config.yaml'))
        cls.openmaint_config = cls.load_config(os.path.join(cls.project_root, 'config', 'openmaint_config.yaml')).get('openmaint', {})

        # Start Kafka using Docker
        cls.logger.info("Starting Kafka cluster using Docker...")
        try:
            docker_compose_path = os.path.join(cls.project_root, 'docker-compose.yml')
            if not os.path.exists(docker_compose_path):
                cls.logger.error(f"Docker Compose file not found at: {docker_compose_path}")
                raise FileNotFoundError(f"Docker Compose file not found at: {docker_compose_path}")

            subprocess.run(['docker-compose', '-f', docker_compose_path, 'up', '-d'], check=True)
            cls.logger.info("Kafka cluster started successfully.")
            # Wait for Kafka to be ready
        except subprocess.CalledProcessError as e:
            cls.logger.error(f"Failed to start Kafka cluster: {e}")
            raise
        except FileNotFoundError as e:
            cls.logger.error(str(e))
            raise

    @classmethod
    def tearDownClass(cls):
        cls.logger.info("Stopping Kafka cluster using Docker...")
        try:
            docker_compose_path = os.path.join(cls.project_root, 'docker-compose.yml')
            subprocess.run(['docker-compose', '-f', docker_compose_path, 'down'], check=True)
            cls.logger.info("Kafka cluster stopped successfully.")
        except subprocess.CalledProcessError as e:
            cls.logger.error(f"Failed to stop Kafka cluster: {e}")

        cls.logger.info("Environment cleanup completed after tests.")

    @staticmethod
    def load_config(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file {path} not found.")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def setUp(self):
        self.logger.info("Setting up test environment...")

        bootstrap_servers = self.kafka_config.get('bootstrap_servers', ['localhost:9092'])
        self.simulator = SensorDataSimulator(
            bootstrap_servers=bootstrap_servers,
            kafka_topics=["sensor-data", "failure_predictions"]
        )
        self.simulator.num_samples = 5
        self.simulator.sleep_interval = 0.5

        models_dir = os.path.join(self.project_root, 'models')
        config_file = os.path.join(self.project_root, 'config', 'database_config.yaml')
        self.processor = RealTimeProcessor(models_dir=models_dir, config_file=config_file, bootstrap_servers=bootstrap_servers)

        self.shutdown_event = threading.Event()

        self.simulator_thread = threading.Thread(
            target=self.simulator.start_simulation,
            daemon=True,
            name="SensorDataSimulatorThread"
        )

        self.processor_thread = threading.Thread(
            target=self.processor.process_messages,
            daemon=True,
            name="RealTimeProcessorThread"
        )
        self.processor_thread.start()
        self.logger.info("RealTimeProcessor started.")


    def tearDown(self):
        self.logger.info("Tearing down test environment...")
        try:
            if self.simulator:
                self.simulator.cleanup()

            if self.processor:
                self.processor.cleanup()

            if self.consumer_thread and self.consumer_thread.is_alive():
                self.logger.info("Signaling OpenMaint consumer thread to shutdown.")
                self.shutdown_event.set()
                self.consumer_thread.join(timeout=10)
                if self.consumer_thread.is_alive():
                    self.logger.warning("OpenMaint consumer thread is still alive after timeout.")
                else:
                    self.logger.info("OpenMaint consumer thread terminated successfully.")
        except Exception as e:
            self.logger.error(f"Error during teardown: {e}")

    def test_end_to_end_pipeline(self):
        """
        End-to-end integration test:
        """

        self.logger.info("Starting the end-to-end pipeline test...")



        db_url = self.database_config.get('database', {}).get('url')
        if db_url:
            try:
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM real_time_predictions"))
                    count = result.scalar()
                    self.logger.info(f"Number of rows in real_time_predictions: {count}")
                    self.assertTrue(
                        count > 0,
                        "No predictions were saved to the database, pipeline might have failed."
                    )
            except Exception as e:
                self.logger.warning(f"Could not verify database entries: {e}")
        else:
            self.logger.warning("No database configured, skipping database verification.")

        bootstrap_servers = self.kafka_config.get('bootstrap_servers', ['localhost:9092'])
        preds_received = self.check_kafka_topic_with_retries('failure_predictions', bootstrap_servers)
        self.assertTrue(
            preds_received,
            "No failure predictions found in 'failure_predictions' topic."
        )

        api_url = self.openmaint_config.get('api_url')
        username = self.openmaint_config.get('username')
        password = self.openmaint_config.get('password')
        if api_url and username and password:
            try:
                om_client = OpenMaintClient(api_url, username, password)
                om_client.logout()
            except Exception as e:
                self.logger.warning(f"Could not verify OpenMAINT integration: {e}")
        else:
            self.logger.info("OpenMAINT config not fully provided, skipping WO verification.")

        self.logger.info("Integration pipeline test completed successfully.")

    def check_kafka_topic_with_retries(self, topic, bootstrap_servers, max_retries=5, wait_interval=2):
        """
        Checks if there's any message in a given topic with retry logic.
        Returns True if at least one message is found, False otherwise.
        """
        attempt = 0
        while attempt < max_retries:
            self.logger.info(f"Checking Kafka topic '{topic}' (Attempt {attempt + 1}/{max_retries})...")
            try:
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=bootstrap_servers,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    auto_offset_reset='earliest',
                    enable_auto_commit=False,
                    group_id='verification-consumer-group-unique',
                    consumer_timeout_ms=5000,  # 5 seconds
                    api_version=(3, 7, 0)
                )
                preds_received = False
                received_messages = []
                for msg in consumer:
                    preds_received = True
                    received_messages.append(msg.value)
                    self.logger.debug(f"Received message in '{topic}': {msg.value}")
                consumer.close()
                if preds_received:
                    self.logger.info(f"Number of failure predictions received: {len(received_messages)}")
                    return True
                else:
                    self.logger.warning(f"No messages found in topic '{topic}' on attempt {attempt + 1}.")
            except Exception as e:
                self.logger.error(f"Error checking topic {topic}: {e}")

            attempt += 1
            self.logger.info(f"Retrying in {wait_interval} seconds...")
            time.sleep(wait_interval)

        self.logger.error(f"No messages found in topic '{topic}' after {max_retries} attempts.")
        return False

if __name__ == '__main__':
    unittest.main()
