# RealTimeProcessing/real_time_processing_runner.py
import os

from RealTimeProcessing.RealTimeProcessorClient import RealTimeProcessor


def main():
    # Load Kafka configurations
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Instantiate the processor
    models_dir = os.path.join(base_dir, 'models')
    config_file = os.path.join(base_dir, 'config', 'database_config.yaml')

    processor = RealTimeProcessor(models_dir=models_dir, config_file=config_file)

    # Start processing messages
    try:
        processor.process_messages()
    except Exception as e:
        processor.logger.error(f"Processor encountered an error: {e}")
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main()
