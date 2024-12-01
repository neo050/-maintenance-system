import os
import subprocess
import time
import pandas as pd
import numpy as np
import logging
import json
from kafka import KafkaProducer
import sys
import socket
import shutil
import psutil

class SensorDataSimulator:
    def __init__(self, kafka_dir, kafka_topics=None, log_file_path='../logs/simulate_sensor_data.log'):
        self.kafka_dir = kafka_dir
        self.kafka_topics = kafka_topics if kafka_topics else ["sensor-data", "failure_predictions"]
        self.kafka_process = None
        self.zookeeper_process = None
        self.producer = None
        self.logger = self.setup_logger(log_file_path)
        self.kafka_log_dir = os.path.join(self.kafka_dir, 'logs')
        self.bootstrap_servers = ['127.0.0.1:9092']
        self.num_samples = 500  # Default number of samples
        self.sleep_interval = 1  # Default sleep time between sending data

    def setup_logger(self, log_file_path):
        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Capture all logs

        # Create handlers
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

        return logger

    def modify_kafka_config(self):
        # Ensure logs directory exists
        logs_dir = os.path.join(self.kafka_dir, 'logs')
        self.ensure_directory_exists(logs_dir)

        config_file = os.path.join(self.kafka_dir, 'config', 'server.properties')
        unique_log_dir = os.path.join(self.kafka_dir, 'logs', f'kafka-logs-{int(time.time())}')
        with open(config_file, 'r') as file:
            lines = file.readlines()
        with open(config_file, 'w') as file:
            for line in lines:
                if line.startswith('log.dirs='):
                    file.write(f'log.dirs={unique_log_dir}\n')
                else:
                    file.write(line)
        self.logger.info(f"Kafka log directory set to: {unique_log_dir}")
        return unique_log_dir

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.logger.info(f"Created directory: {directory}")

    def terminate_processes(self, process_names):
        """Terminate processes with the given names."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline_list = proc.info['cmdline']
                if cmdline_list and isinstance(cmdline_list, list):
                    cmdline = ' '.join(cmdline_list)
                else:
                    cmdline = ''
                if any(name.lower() in cmdline.lower() for name in process_names):
                    self.logger.info(f"Terminating process {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.terminate()
                    proc.wait(timeout=30)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    def delete_log_directories(self, log_dirs):
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                try:
                    shutil.rmtree(log_dir)
                    self.logger.info(f"Deleted log directory: {log_dir}")
                except Exception as e:
                    self.logger.error(f"Failed to delete log directory {log_dir}: {e}")

    def is_port_in_use(self, port):
        """Check if a port is in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        except Exception as e:
            self.logger.error(f"Error checking for running processes: {e}")
            return False

    def wait_for_zookeeper_ready(self, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection(('localhost', 2181), timeout=30):
                    self.logger.info("Zookeeper is ready.")
                    return True
            except Exception:
                self.logger.info("Waiting for Zookeeper to be ready...")
                time.sleep(5)
        self.logger.error("Zookeeper did not become ready in time.")
        return False

    def start_zookeeper(self):
        try:
            if self.is_port_in_use(2181):
                self.logger.info("Zookeeper is already running.")
                if not self.wait_for_zookeeper_ready():
                    raise Exception("Zookeeper is not ready.")
                return None
            self.logger.info("Starting Zookeeper...")

            # Ensure logs directory exists
            logs_dir = os.path.join(self.kafka_dir, 'logs')
            self.ensure_directory_exists(logs_dir)

            zookeeper_cmd = os.path.join(self.kafka_dir, 'bin', 'windows', 'zookeeper-server-start.bat')
            zookeeper_config = os.path.join(self.kafka_dir, 'config', 'zookeeper.properties')
            zookeeper_stdout = open(os.path.join(self.kafka_dir, 'logs', 'zookeeper_stdout.log'), 'w')
            zookeeper_stderr = open(os.path.join(self.kafka_dir, 'logs', 'zookeeper_stderr.log'), 'w')
            self.zookeeper_process = subprocess.Popen(
                [zookeeper_cmd, zookeeper_config],
                stdout=zookeeper_stdout,
                stderr=zookeeper_stderr
            )
            if not self.wait_for_zookeeper_ready():
                self.zookeeper_process.terminate()
                raise Exception("Zookeeper did not start in time.")
            self.logger.info("Zookeeper started.")
        except Exception as e:
            self.logger.exception("Failed to start Zookeeper")
            raise

    def wait_for_kafka_ready(self, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection(('localhost', 9092), timeout=30):
                    self.logger.info("Kafka broker is ready.")
                    return True
            except Exception:
                self.logger.info("Waiting for Kafka broker to be ready...")
                time.sleep(5)
        self.logger.error("Kafka broker did not become ready in time.")
        return False

    def start_kafka_broker(self):
        try:
            kafka_log_dir = self.modify_kafka_config()
            if self.is_port_in_use(9092):
                self.logger.info("Kafka broker is already running.")
                if not self.wait_for_kafka_ready():
                    raise Exception("Kafka broker is not ready.")
                return None

            self.logger.info("Starting Kafka broker...")
            kafka_cmd = os.path.join(self.kafka_dir, 'bin', 'windows', 'kafka-server-start.bat')
            kafka_config = os.path.join(self.kafka_dir, 'config', 'server.properties')
            kafka_stdout = open(os.path.join(self.kafka_dir, 'logs', 'kafka_stdout.log'), 'w')
            kafka_stderr = open(os.path.join(self.kafka_dir, 'logs', 'kafka_stderr.log'), 'w')
            # Use --override to set log.dirs
            self.kafka_process = subprocess.Popen(
                [kafka_cmd, kafka_config, '--override', f'log.dirs={kafka_log_dir}'],
                stdout=kafka_stdout,
                stderr=kafka_stderr
            )
            if not self.wait_for_kafka_ready():
                self.kafka_process.terminate()
                raise Exception("Kafka broker did not start in time.")
            self.logger.info("Kafka broker started.")
        except Exception as e:
            self.logger.exception("Failed to start Kafka broker")
            raise

    def create_kafka_topic(self, topic_name):
        try:
            self.logger.info(f"Creating Kafka topic '{topic_name}'...")
            kafka_topics_cmd = os.path.join(self.kafka_dir, 'bin', 'windows', 'kafka-topics.bat')
            command = f'"{kafka_topics_cmd}" --create --topic {topic_name} --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1'
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                stderr = result.stderr
                if 'TopicExistsException' in stderr:
                    self.logger.info(f"Kafka topic '{topic_name}' already exists.")
                else:
                    self.logger.error(f"Failed to create Kafka topic: {stderr}")
                    raise Exception(f"Failed to create Kafka topic: {stderr}")
            else:
                self.logger.info(f"Kafka topic '{topic_name}' created.")
        except Exception as e:
            self.logger.error(f"Failed to create Kafka topic: {e}")
            raise

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
            raise

    def send_data(self, topic, data):
        try:
            # Log the data being sent
            self.logger.debug(f"Sending data to Kafka: {data}")
            self.producer.send(topic, value=data)
            self.producer.flush()
            self.logger.info(f"Data sent to Kafka: {data}")
        except Exception as e:
            self.logger.error(f"Failed to send data: {e}")

    def simulate_sensor_data(self):
        num_samples = self.num_samples
        # Set random seed for reproducibility
        np.random.seed(42)

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

        # Heat Dissipation Failure (HDF)
        hdf_indices = np.where((process_temp - air_temp < 8.6) & (rotational_speed < 1380))[0]
        hdf[hdf_indices] = 1
        machine_failure[hdf_indices] = 1

        # Power Failure (PWF)
        power = torque * (rotational_speed * 2 * np.pi / 60)  # Power in watts (rad/s * torque)
        pwf_indices = np.where((power < 3500) | (power > 9000))[0]
        pwf[pwf_indices] = 1
        machine_failure[pwf_indices] = 1

        # Overstrain Failure (OSF)
        osf_threshold = {'L': 11000, 'M': 12000, 'H': 13000}
        osf_indices = [i for i, p in enumerate(product_types) if tool_wear[i] * torque[i] > osf_threshold[p]]
        osf[osf_indices] = 1
        machine_failure[osf_indices] = 1

        # Random Failures (RNF)
        rnf_indices = np.random.choice(num_samples, size=max(1, int(num_samples * 0.001)), replace=False)
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

        return synthetic_data

    def start_simulation(self):
        try:
            # Delete Kafka and Zookeeper log directories
            self.delete_log_directories([self.kafka_log_dir])

            # Terminate existing Zookeeper and Kafka processes
            self.terminate_processes(['zookeeper-server-start', 'kafka-server-start'])

            # Start Zookeeper and Kafka broker
            while True:
                try:
                    self.start_zookeeper()
                    self.start_kafka_broker()
                    break
                except Exception as e:
                    self.logger.error(f"An error occurred while starting services: {e}")
                    time.sleep(5)  # Wait before retrying

            # Create Kafka topics
            for topic in self.kafka_topics:
                self.create_kafka_topic(topic)

            # Create Kafka producer
            self.create_producer()

            # Simulate sensor data
            simulated_data = self.simulate_sensor_data()

            # Send data to Kafka
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
                time.sleep(self.sleep_interval)  # Adjust the sleep time as needed

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received. Exiting gracefully.")

        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        # Close the Kafka producer
        if self.producer:
            self.logger.info("Closing the Kafka producer.")
            self.producer.close()

        # Shutdown Kafka and Zookeeper
        self.logger.info("Shutting down Kafka broker and Zookeeper...")
        if self.kafka_process:
            self.kafka_process.terminate()
            try:
                self.kafka_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.logger.warning("Kafka process did not terminate in time. Killing it.")
                self.kafka_process.kill()

        if self.zookeeper_process:
            self.zookeeper_process.terminate()
            try:
                self.zookeeper_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.logger.warning("Zookeeper process did not terminate in time. Killing it.")
                self.zookeeper_process.kill()

        self.logger.info("Kafka broker and Zookeeper have been shut down.")
