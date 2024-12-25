import os
import sys
import time
import logging
import subprocess
import webbrowser
import yaml
import zipfile
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
TF_ENABLE_ONEDNN_OPTS = 0
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

def extract_zip_file(zip_path, extract_to, logger):
    """
    Extracts the contents of the given zip file to the specified directory.
    If the directory does not exist, it will be created.
    """
    if not os.path.exists(zip_path):
        logger.info(f"No ZIP file found at {zip_path}. Skipping extraction.")
        return False

    try:
        logger.info(f"Extracting {zip_path} to {extract_to}...")
        if not os.path.exists(extract_to):
            os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extraction of {zip_path} completed successfully.")
        return True
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file {zip_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        raise

def start_services(docker_compose_path, service_desc, logger, wait_time=20):
    if not os.path.exists(docker_compose_path):
        logger.error(f"Docker Compose file not found at: {docker_compose_path}")
        raise FileNotFoundError(f"Docker Compose file not found at: {docker_compose_path}")

    logger.info(f"Starting {service_desc} using Docker Compose...")
    try:
        subprocess.run(['docker-compose', '-f', docker_compose_path, 'up', '-d'], check=True)
        logger.info(f"{service_desc} started successfully. Waiting {wait_time} seconds for services to be ready.")
        time.sleep(wait_time)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start {service_desc}: {e}")
        raise
    except KeyboardInterrupt as e:
        logger.error("An KeyboardInterrupt interrupt  while running processes.", exc_info=True)
        sys.exit(0)

    except Exception as e:
        logger.error(f"An Exception interrupted  while running processes.{e}", exc_info=True)
        raise

def main():


    # Use the directory where run.py is located as the base directory
    base_dir = os.path.abspath(os.path.dirname(__file__))
    logger.debug(f"Base directory: {base_dir}")

    # Paths to docker-compose files
    kafka_compose_path = os.path.join(base_dir, 'docker-compose.yml')
    openmaint_compose_path = os.path.join(base_dir, 'IntegrationWithExistingSystems', 'openmaint-2.3-3.4.1-d',
                                          'docker-compose.yml')

    # Construct paths to config files relative to base_dir
    openmaint_config_path = os.path.join(base_dir, 'config', 'openmaint_config.yaml')
    grafana_config_path = os.path.join(base_dir, 'config', 'grafana_config.yaml')

    # Before loading configs, extract any required ZIP files.
    # Let's assume the ZIP files for openMAINT and Grafana are located similarly:
    # e.g., openmaint-2.3-3.4.1-d.zip for openMAINT and maybe some grafana-related zip if needed.
    # Adjust these names/paths as required by your project.
    openmaint_zip_path = os.path.join(base_dir, 'IntegrationWithExistingSystems', 'openmaint-2.3-3.4.1-d.zip')
    openmaint_extract_path = os.path.join(base_dir, 'IntegrationWithExistingSystems')

    kafka_zip_path = os.path.join(base_dir, 'RealTimeProcessing', 'kafka.zip')
    kafka_extract_path = os.path.join(base_dir, 'RealTimeProcessing')

    # If there are other zip files to extract, replicate the pattern:
    # grafana_zip_path = os.path.join(base_dir, 'grafana.zip')
    # grafana_extract_path = os.path.join(base_dir, 'grafana_extracted_folder')

    if not os.path.exists(openmaint_compose_path):
        # Extract openmaint zip if exists
        try:
            extract_zip_file(openmaint_zip_path, openmaint_extract_path, logger)
        except Exception as e:
            logger.error("Failed to extract openmaint ZIP file.", exc_info=True)
            sys.exit(1)

    if not os.path.exists(os.path.join(base_dir, 'RealTimeProcessing', 'kafka')):
        try:
            extract_zip_file(kafka_zip_path, kafka_extract_path, logger)
        except Exception as e:
            logger.error("Failed to extract kafka ZIP file.", exc_info=True)
            sys.exit(1)

    # Now load configurations
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
        logger.info("OpenMaint's: Username: admin  Password: admin")
        logger.info(f"Grafana URL: {grafana_url}")
    except KeyError as e:
        logger.error(f"Missing expected keys in configuration: {e}", exc_info=True)
        sys.exit(1)



    # Start Kafka cluster
    try:
        start_services(kafka_compose_path, "Kafka cluster", logger, wait_time=15)#15
    except Exception as e:
        logger.error("Could not start Kafka services.", exc_info=True)
        sys.exit(1)

    # Start openMAINT cluster
    try:
        start_services(openmaint_compose_path, "openMAINT cluster", logger, wait_time=45)#45
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
        processes.append(subprocess.Popen([sys.executable, sensor_script]))
        logger.info("Starting RealTimeProcessor...")
        processes.append(subprocess.Popen([sys.executable, processor_script]))

        logger.info("Starting openmaint_consumer...")
        processes.append(subprocess.Popen([sys.executable, openmaint_script]))

        # Wait a bit before opening browsers
        time.sleep(10)

        # Open the URLs in the default browser
        logger.info("Opening configured URLs in the default browser...")
        try:
            webbrowser.open(openmaint_url)
            webbrowser.open(grafana_url)
        except Exception as e:
            logger.error(f"Failed to open URLs in the browser: {e}", exc_info=True)

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
    logger = setup_logging()
    try:
        main()
    except Exception as e:
        logger.error(f"An Exception interrupted  while running processes.{e}", exc_info=True)


