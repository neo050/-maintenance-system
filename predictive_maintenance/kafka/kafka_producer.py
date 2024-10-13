from kafka import KafkaProducer
import json
import yaml

def load_kafka_config(config_file='../config/kafka_config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_producer(config):
    producer = KafkaProducer(
        bootstrap_servers=config['bootstrap_servers'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        acks=config['producer_settings']['acks'],
        retries=config['producer_settings']['retries']
    )
    return producer

def send_data(producer, topic, data):
    producer.send(topic, data)
    producer.flush()

if __name__ == "__main__":
    config = load_kafka_config()
    producer = create_producer(config)
    topic = config['topics'][0]  # 'sensor-data'
    # Example data to send
    data = {'sensor_id': 1, 'value': 100}
    send_data(producer, topic, data)
