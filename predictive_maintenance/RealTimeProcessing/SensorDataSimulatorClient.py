# RealTimeProcessing/SensorDataSimulatorClient.py

import os
import pandas as pd
import numpy as np
import logging
import json
from kafka import KafkaProducer
import sys
import socket
import subprocess
import time

class SensorDataSimulator:
    def __init__(self, bootstrap_servers=['localhost:9092'], kafka_topics=None, log_file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'logs','simulate_sensor_data.log')):
        self.logger = self.setup_logger(log_file_path)

        self.logger.info("Starting Kafka cluster using Docker...")

        try:
            docker_compose_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                                               'docker-compose.yml')
            if not os.path.exists(docker_compose_path):
                self.logger.error(f"Docker Compose file not found at: {docker_compose_path}")
                raise FileNotFoundError(f"Docker Compose file not found at: {docker_compose_path}")

            subprocess.run(['docker-compose', '-f', docker_compose_path, 'up', '-d'], check=True)
            self.logger.info("Kafka cluster started successfully.")
            # Wait for Kafka to be ready
            time.sleep(10)  # Increased wait time to ensure Kafka is ready

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start Kafka cluster: {e}")
            raise
        except FileNotFoundError as e:
            self.logger.error(str(e))
            raise
        except KeyboardInterrupt as e:
            self.logger.error(f"Simulation encountered an error: KeyboardInterrupt  {e} end of simulation ")
            return
        except Exception as e:
            self.logger.error(str(e))
            raise

        self.kafka_topics = kafka_topics if kafka_topics else ["sensor-data", "failure_predictions"]
        self.producer = None
        self.bootstrap_servers = bootstrap_servers
        self.num_samples = 500
        self.sleep_interval = 1

    def setup_logger(self, log_file_path):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

        return logger

    def create_producer(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                api_version=(3, 7, 0),
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            self.logger.info("Kafka producer created.")
        except Exception as e:
            self.logger.error(f"Failed to create Kafka producer: {e}")
            raise Exception("Failed to create Kafka producer: " + str(e))

    def send_data(self, topic, data):
        try:
            self.logger.debug(f"Sending data to Kafka: {data}")
            self.producer.send(topic, value=data)
            self.producer.flush()
            self.logger.info(f"Data sent to Kafka: {data}")
        except Exception as e:
            self.logger.error(f"Failed to send data: {e}")

    def simulate_sensor_data(self):
        num_samples = self.num_samples
        if num_samples == 0:
            return pd.DataFrame(columns=[
                'UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure',
                'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
            ])

        np.random.seed(42)
        product_types = np.random.choice(['L', 'M', 'H'], size=num_samples, p=[0.5, 0.3, 0.2])
        product_ids = [f'{ptype}{np.random.randint(10000, 99999)}' for ptype in product_types]

        air_temp = np.random.normal(loc=300, scale=2, size=num_samples)
        process_temp = air_temp + 10 + np.random.normal(loc=0, scale=1, size=num_samples)
        rotational_speed = np.random.normal(loc=1500, scale=100, size=num_samples)
        torque = np.clip(np.random.normal(loc=40, scale=10, size=num_samples), a_min=0, a_max=None)
        tool_wear = np.array([2 if p == 'L' else 3 if p == 'M' else 5 for p in product_types])
        tool_wear = tool_wear + np.random.randint(0, 240, size=num_samples)

        machine_failure = np.zeros(num_samples, dtype=int)
        twf = np.zeros(num_samples, dtype=int)
        hdf = np.zeros(num_samples, dtype=int)
        pwf = np.zeros(num_samples, dtype=int)
        osf = np.zeros(num_samples, dtype=int)
        rnf = np.zeros(num_samples, dtype=int)

        twf_indices = np.where((tool_wear >= 200) & (tool_wear <= 240))[0]
        twf[twf_indices] = 1
        machine_failure[twf_indices] = np.random.choice([0, 1], size=len(twf_indices), p=[0.43, 0.57])

        hdf_indices = np.where((process_temp - air_temp < 8.6) & (rotational_speed < 1380))[0]
        hdf[hdf_indices] = 1
        machine_failure[hdf_indices] = 1

        power = torque * (rotational_speed * 2 * np.pi / 60)
        pwf_indices = np.where((power < 3500) | (power > 9000))[0]
        pwf[pwf_indices] = 1
        machine_failure[pwf_indices] = 1

        osf_threshold = {'L': 11000, 'M': 12000, 'H': 13000}
        osf_indices = [i for i, p in enumerate(product_types) if tool_wear[i] * torque[i] > osf_threshold[p]]
        for idx in osf_indices:
            osf[idx] = 1
            machine_failure[idx] = 1

        rnf_indices = np.random.choice(num_samples, size=max(1, int(num_samples * 0.001)), replace=False)
        rnf[rnf_indices] = 1
        machine_failure[rnf_indices] = 1

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
        return synthetic_data

    def start_simulation(self):
        try:
            # Ensure Kafka is running
            if not self.is_kafka_running():
                self.logger.error("Kafka is not running. Cannot start simulation.")
                return

            # Create necessary Kafka topics
            for topic in self.kafka_topics:
                self.create_kafka_topic(topic)

            self.create_producer()
            simulated_data = self.simulate_sensor_data()

            for index, row in simulated_data.iterrows():
                data = {
                    'Air temperature [K]': row['Air temperature [K]'],
                    'Process temperature [K]': row['Process temperature [K]'],
                    'Rotational speed [rpm]': row['Rotational speed [rpm]'],
                    'Torque [Nm]': row['Torque [Nm]'],
                    'Tool wear [min]': row['Tool wear [min]'],
                    'Type': row['Type'],
                    'UDI': row["UDI"],
                    'Product ID': row["Product ID"],
                    'Machine failure': row['Machine failure'],
                    'TWF': row['TWF'],
                    'HDF': row['HDF'],
                    'PWF': row['PWF'],
                    'OSF': row['OSF'],
                    'RNF': row['RNF']
                }
                self.send_data('sensor-data', data)
                time.sleep(self.sleep_interval)

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received. Exiting gracefully.")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
        finally:
            self.cleanup()

    def is_kafka_running(self):
        try:
            with socket.create_connection(('localhost', 9092), timeout=5):
                self.logger.info("Kafka is running.")
                return True
        except Exception:
            self.logger.error("Kafka is not reachable on localhost:9092.")
            return False

    def create_kafka_topic(self, topic_name):
        self.logger.info(f"Creating Kafka topic '{topic_name}'...")
        try:
            # Use docker exec to create topic within the Dockerized Kafka
            subprocess.run([
                'docker', 'exec', 'kafka', 'kafka-topics', '--create',
                '--topic', topic_name,
                '--bootstrap-server', 'localhost:9092',
                '--replication-factor', '1',
                '--partitions', '1'
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.logger.info(f"Kafka topic '{topic_name}' created.")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode()
            output = e.stdout.decode()
            if 'TopicExistsException' in stderr or 'TopicExistsException' in output:
                self.logger.info(f"Kafka topic '{topic_name}' already exists.")
            else:
                self.logger.error(f"Failed to create Kafka topic: {stderr}")
                raise Exception(f"Failed to create Kafka topic: {stderr}")

    def cleanup(self):
        try:
            if self.producer:
                self.logger.info("Closing the Kafka producer.")
                self.producer.close()
        except Exception as e:
            self.logger.error(f"Error while closing producer: {e}")

        try:
            # Log before closing handlers
            self.logger.info("SensorDataSimulatorClient is shutting down.")

            # Close and remove all handlers associated with the logger
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)
                self.logger.debug(f"Closed and removed handler: {handler}")
        except Exception as e:
            print(f"Error while closing logger: {e}")
