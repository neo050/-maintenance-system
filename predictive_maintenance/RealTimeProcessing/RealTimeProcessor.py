# RealTimeProcessing/real_time_processing_runner.py
import os

from RealTimeProcessorClient import RealTimeProcessor


def main():
    # Load Kafka configurations
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(base_dir, 'models')
    config_file = os.path.join(base_dir, 'config', 'database_config.yaml')

    processor = None  # <--- Initialize to None
    try:
        processor = RealTimeProcessor(models_dir=models_dir, config_file=config_file)
        processor.process_messages()
    except Exception as e:
        # If processor is None, we can't use processor.logger yet
        if processor is not None:
            processor.logger.error(f"Processor encountered an error: {e}")
        else:
            # Fallback: use normal logging
            import logging
            logging.error(f"Processor failed to initialize: {e}")
    finally:
        if processor is not None:
            processor.cleanup()



if __name__ == "__main__":
    main()
