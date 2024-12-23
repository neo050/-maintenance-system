# cleanup_environment.py
import shutil
import subprocess
import time
import os
import signal
import logging
import sys

def setup_logger():
    logger = logging.getLogger("CleanupEnvironment")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

def terminate_docker_containers(logger):
    try:
        logger.info("Stopping and removing Docker containers...")
        subprocess.run(['docker-compose', '-f', 'docker-compose.test.yml', 'down'], check=True)
        logger.info("Docker containers stopped and removed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error stopping Docker containers: {e}")

def delete_kafka_logs(logger):
    kafka_log_dir = os.path.abspath('RealTimeProcessing/kafka/kafka_2.13-3.7.0/logs')
    if os.path.exists(kafka_log_dir):
        try:
            logger.info(f"Deleting Kafka log directory: {kafka_log_dir}")
            shutil.rmtree(kafka_log_dir)
            logger.info("Kafka log directory deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting Kafka log directory: {e}")
    else:
        logger.info("Kafka log directory does not exist. No action needed.")

def main():
    logger = setup_logger()
    terminate_docker_containers(logger)
    delete_kafka_logs(logger)
    logger.info("Environment cleanup completed.")

if __name__ == "__main__":
    main()
