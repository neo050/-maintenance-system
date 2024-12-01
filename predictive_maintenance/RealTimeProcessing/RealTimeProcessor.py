# real_time_processing_runner.py
from RealTimeProcessorClient import RealTimeProcessor

def main():
    models_dir = '../models'
    config_file = '../config/database_config.yaml'
    processor = RealTimeProcessor(models_dir, config_file)
    processor.process_messages()

if __name__ == "__main__":
    main()
