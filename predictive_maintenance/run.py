import os
import sys
import time
import logging
import subprocess
import webbrowser
import yaml

def setup_logging(log_level=logging.DEBUG):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(ch)

    return logger

def load_yaml_config(file_path, logger):
    if not os.path.exists(file_path):
        logger.error(f"Configuration file does not exist: {file_path}")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error loading YAML config from {file_path}: {e}")
            raise

def start_services(docker_compose_path, service_desc, logger, wait_time=20):
    if not os.path.exists(docker_compose_path):
        logger.error(f"Docker Compose file not found at: {docker_compose_path}")
        raise FileNotFoundError(f"Docker Compose file not found at: {docker_compose_path}")

    logger.info(f"Starting {service_desc} using Docker Compose...")
    try:
        subprocess.run(['docker-compose', '-f', docker_compose_path, 'up', '-d'], check=True)
        logger.info(f"{service_desc} started successfully.")
        time.sleep(wait_time)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start {service_desc}: {e}")
        raise

def main():
    logger = setup_logging()

    # Use the directory where run.py is located as the base directory
    base_dir = os.path.abspath(os.path.dirname(__file__))
    logger.debug(f"Base directory: {base_dir}")

    # Construct paths to config files relative to base_dir
    openmaint_config_path = os.path.join(base_dir, 'config', 'openmaint_config.yaml')
    grafana_config_path = os.path.join(base_dir, 'config', 'grafana_config.yaml')

    # Load configurations
    try:
        openmaint_config = load_yaml_config(openmaint_config_path, logger)
        grafana_config = load_yaml_config(grafana_config_path, logger)
    except Exception as e:
        logger.error("Failed to load configuration files.", exc_info=True)
        sys.exit(1)

    # Extract URLs
    try:
        openmaint_url = openmaint_config['openmaint']['work_order']
        grafana_url = grafana_config['url']
        logger.info(f"OpenMaint work_order URL: {openmaint_url}")
        logger.info(f"Grafana URL: {grafana_url}")
    except KeyError as e:
        logger.error(f"Missing expected keys in configuration: {e}", exc_info=True)
        sys.exit(1)

    # Paths to docker-compose files (adjust these if your docker-compose files are elsewhere)
    kafka_compose_path = os.path.join(base_dir, 'docker-compose.yml')
    openmaint_compose_path = os.path.join(base_dir, 'IntegrationWithExistingSystems', 'openmaint-2.3-3.4.1-d', 'docker-compose.yml')

    # Start Kafka cluster
    try:
        start_services(kafka_compose_path, "Kafka cluster", logger, wait_time=15)
    except Exception as e:
        logger.error("Could not start Kafka services.", exc_info=True)
        sys.exit(1)

    # Start openMAINT cluster
    try:
        start_services(openmaint_compose_path, "openMAINT cluster", logger, wait_time=45)
    except Exception as e:
        logger.error("Could not start openMAINT services.", exc_info=True)
        sys.exit(1)

    # Paths to scripts
    sensor_script = os.path.join(base_dir, 'RealTimeProcessing', 'SensorDataSimulator.py')
    processor_script = os.path.join(base_dir, 'RealTimeProcessing', 'RealTimeProcessor.py')
    openmaint_script = os.path.join(base_dir, 'IntegrationWithExistingSystems', 'openmaint_consumer.py')

    for script_path in [sensor_script, processor_script, openmaint_script]:
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            sys.exit(1)

    processes = []
    try:
        logger.info("Starting SensorDataSimulator...")
        processes.append(subprocess.Popen(['python', sensor_script]))

        logger.info("Starting RealTimeProcessor...")
        processes.append(subprocess.Popen(['python', processor_script]))

        logger.info("Starting openmaint_consumer...")
        processes.append(subprocess.Popen(['python', openmaint_script]))

        time.sleep(10)

        # Open the URLs in the default browser
        logger.info("Opening configured URLs in the default browser...")
        webbrowser.open(openmaint_url)
        webbrowser.open(grafana_url)

        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Terminating all processes.")
        for p in processes:
            if p.poll() is None:
                p.terminate()
    except Exception as e:
        logger.error("An unexpected error occurred while running processes.", exc_info=True)
        for p in processes:
            if p.poll() is None:
                p.terminate()
        sys.exit(1)
    finally:
        logger.info("All processes have completed or been terminated.")

    logger.info("Script completed successfully.")

if __name__ == "__main__":
    main()
