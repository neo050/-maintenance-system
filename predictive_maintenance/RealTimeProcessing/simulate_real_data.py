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

# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all logs

# Create handlers
log_file_path = '../logs/simulate_sensor_data.log'
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

def modify_kafka_config(kafka_dir=r"C:\kafka\kafka_2.13-3.7.0"):
    # Ensure logs directory exists
    logs_dir = os.path.join(kafka_dir, 'logs')
    ensure_directory_exists(logs_dir)

    config_file = os.path.join(kafka_dir, 'config', 'server.properties')
    unique_log_dir = os.path.join(kafka_dir, 'logs', f'kafka-logs-{int(time.time())}')
    with open(config_file, 'r') as file:
        lines = file.readlines()
    with open(config_file, 'w') as file:
        for line in lines:
            if line.startswith('log.dirs='):
                file.write(f'log.dirs={unique_log_dir}\n')
            else:
                file.write(line)
    logger.info(f"Kafka log directory set to: {unique_log_dir}")
    return unique_log_dir


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
def terminate_processes(process_names):
    """Terminate processes with the given names."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline_list = proc.info['cmdline']
            if cmdline_list and isinstance(cmdline_list, list):
                cmdline = ' '.join(cmdline_list)
            else:
                cmdline = ''
            if any(name.lower() in cmdline.lower() for name in process_names):
                logger.info(f"Terminating process {proc.info['name']} (PID: {proc.info['pid']})")
                proc.terminate()
                proc.wait(timeout=30)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


def delete_log_directories(log_dirs):
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            try:
                shutil.rmtree(log_dir)
                logger.info(f"Deleted log directory: {log_dir}")
            except Exception as e:
                logger.error(f"Failed to delete log directory {log_dir}: {e}")


def is_port_in_use(port):
    """Check if a port is in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except Exception as e:
        logger.error(f"Error checking for running processes: {e}")
        return False


def wait_for_zookeeper_ready(timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(('localhost', 2181), timeout=30):
                logger.info("Zookeeper is ready.")
                return True
        except Exception:
            logger.info("Waiting for Zookeeper to be ready...")
            time.sleep(5)
    logger.error("Zookeeper did not become ready in time.")
    return False


def start_zookeeper(kafka_dir):
    try:
        if is_port_in_use(2181):
            logger.info("Zookeeper is already running.")
            if not wait_for_zookeeper_ready():
                raise Exception("Zookeeper is not ready.")
            return None
        logger.info("Starting Zookeeper...")

        # Ensure logs directory exists
        logs_dir = os.path.join(kafka_dir, 'logs')
        ensure_directory_exists(logs_dir)

        zookeeper_cmd = os.path.join(kafka_dir, 'bin', 'windows', 'zookeeper-server-start.bat')
        zookeeper_config = os.path.join(kafka_dir, 'config', 'zookeeper.properties')
        zookeeper_stdout = open(os.path.join(kafka_dir, 'logs', 'zookeeper_stdout.log'), 'w')
        zookeeper_stderr = open(os.path.join(kafka_dir, 'logs', 'zookeeper_stderr.log'), 'w')
        zookeeper_process = subprocess.Popen(
            [zookeeper_cmd, zookeeper_config],
            stdout=zookeeper_stdout,
            stderr=zookeeper_stderr
        )
        if not wait_for_zookeeper_ready():
            zookeeper_process.terminate()
            raise Exception("Zookeeper did not start in time.")
        logger.info("Zookeeper started.")
        return zookeeper_process
    except Exception as e:
        logger.exception("Failed to start Zookeeper")
        raise


def wait_for_kafka_ready(timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(('localhost', 9092), timeout=30):
                logger.info("Kafka broker is ready.")
                return True
        except Exception:
            logger.info("Waiting for Kafka broker to be ready...")
            time.sleep(5)
    logger.error("Kafka broker did not become ready in time.")
    return False


def start_kafka_broker(kafka_dir, kafka_log_dir=modify_kafka_config()):
    try:
        if is_port_in_use(9092):
            logger.info("Kafka broker is already running.")
            if not wait_for_kafka_ready():
                raise Exception("Kafka broker is not ready.")
            return None

            # Ensure logs directory exists
            logs_dir = os.path.join(kafka_dir, 'logs')
            ensure_directory_exists(logs_dir)

        logger.info("Starting Kafka broker...")
        kafka_cmd = os.path.join(kafka_dir, 'bin', 'windows', 'kafka-server-start.bat')
        kafka_config = os.path.join(kafka_dir, 'config', 'server.properties')
        kafka_stdout = open(os.path.join(kafka_dir, 'logs', 'kafka_stdout.log'), 'w')
        kafka_stderr = open(os.path.join(kafka_dir, 'logs', 'kafka_stderr.log'), 'w')
        # Use --override to set log.dirs
        kafka_process = subprocess.Popen(
            [kafka_cmd, kafka_config, '--override', f'log.dirs={kafka_log_dir}'],
            stdout=kafka_stdout,
            stderr=kafka_stderr
        )
        if not wait_for_kafka_ready():
            kafka_process.terminate()
            raise Exception("Kafka broker did not start in time.")
        logger.info("Kafka broker started.")
        return kafka_process
    except Exception as e:
        logger.exception("Failed to start Kafka broker")
        raise


def create_kafka_topic(kafka_dir, topic_name):
    try:
        logger.info(f"Creating Kafka topic '{topic_name}'...")
        kafka_topics_cmd = os.path.join(kafka_dir, 'bin', 'windows', 'kafka-topics.bat')
        command = f'"{kafka_topics_cmd}" --create --topic {topic_name} --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1'
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            stderr = result.stderr
            if 'TopicExistsException' in stderr:
                logger.info(f"Kafka topic '{topic_name}' already exists.")
            else:
                logger.error(f"Failed to create Kafka topic: {stderr}")
                raise Exception(f"Failed to create Kafka topic: {stderr}")
        else:
            logger.info(f"Kafka topic '{topic_name}' created.")
    except Exception as e:
        logger.error(f"Failed to create Kafka topic: {e}")
        raise


def create_producer(bootstrap_servers):
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            api_version=(3, 7, 0),
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        logger.info("Kafka producer created.")
        return producer
    except Exception as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        raise


def send_data(producer, topic, data):
    try:
        # Log the data being sent
        logger.debug(f"Sending data to Kafka: {data}")
        producer.send(topic, value=data)
        producer.flush()
        logger.info(f"Data sent to Kafka: {data}")
    except Exception as e:
        logger.error(f"Failed to send data: {e}")


def simulate_sensor_data(num_samples):
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

    # Heat Dissipation Failure (HDF): difference between air and process temperature < 8.6 K and rotational speed < 1380 rpm
    hdf_indices = np.where((process_temp - air_temp < 8.6) & (rotational_speed < 1380))[0]
    hdf[hdf_indices] = 1
    machine_failure[hdf_indices] = 1

    # Power Failure (PWF): power (torque * rotational speed) out of range [3500, 9000 W]
    power = torque * (rotational_speed * 2 * np.pi / 60)  # Power in watts (rad/s * torque)
    pwf_indices = np.where((power < 3500) | (power > 9000))[0]
    pwf[pwf_indices] = 1
    machine_failure[pwf_indices] = 1

    # Overstrain Failure (OSF): tool wear * torque exceeds threshold based on product type
    osf_threshold = {'L': 11000, 'M': 12000, 'H': 13000}
    osf_indices = [i for i, p in enumerate(product_types) if tool_wear[i] * torque[i] > osf_threshold[p]]
    osf[osf_indices] = 1
    machine_failure[osf_indices] = 1

    # Random Failures (RNF): 0.1% chance of failure regardless of conditions
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




def main():
    # Specify the path to your Kafka installation directory
    kafka_dir = r"C:\kafka\kafka_2.13-3.7.0"

    # Delete Kafka and Zookeeper log directories
    kafka_log_dir = r"C:\kafka\kafka_2.13-3.7.0\logs"
    delete_log_directories([kafka_log_dir])

    # Terminate existing Zookeeper and Kafka processes
    terminate_processes(['zookeeper-server-start', 'kafka-server-start'])

    while True:
        try:
            # Start Zookeeper and Kafka broker
            zookeeper_process = start_zookeeper(kafka_dir)
            kafka_process = start_kafka_broker(kafka_dir)
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    # Create Kafka topics
    topics = ["sensor-data", "failure_predictions"]
    for topic in topics:
        create_kafka_topic(kafka_dir, topic)

    # Create Kafka producer
    bootstrap_servers = ['127.0.0.1:9092']
    producer = create_producer(bootstrap_servers)

    try:
        # Simulate sensor data
        num_samples = 500  # Adjust the number of samples as needed
        simulated_data = simulate_sensor_data(num_samples)

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
            send_data(producer, 'sensor-data', data)
            time.sleep(1)  # Adjust the sleep time as needed

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting gracefully.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


    finally:
        # Close the Kafka producer
        if 'producer' in locals() and producer:
            logger.info("Closing the Kafka producer.")
            producer.close()

        # Shutdown Kafka and Zookeeper
        logger.info("Shutting down Kafka broker and Zookeeper...")
        if 'kafka_process' in locals() and kafka_process:
            kafka_process.terminate()
            try:
                kafka_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Kafka process did not terminate in time. Killing it.")
                kafka_process.kill()

        if 'zookeeper_process' in locals() and zookeeper_process:
            zookeeper_process.terminate()
            try:
                zookeeper_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Zookeeper process did not terminate in time. Killing it.")
                zookeeper_process.kill()

        logger.info("Kafka broker and Zookeeper have been shut down.")


if __name__ == "__main__":
    main()
