from kafka import KafkaProducer
import json
import pandas as pd

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def send_data():
    data = pd.read_csv('../data/processed/processed_data_with_features.csv')
    for index, row in data.iterrows():
        producer.send('sensor_topic', row.to_dict())

if __name__ == "__main__":
    send_data()
