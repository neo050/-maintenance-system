import time
import random
from kafka import KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def generate_sensor_data():
    data = {
        'sensor_id': random.randint(1, 100),
        'temperature': random.uniform(20.0, 25.0),
        'pressure': random.uniform(1.0, 2.0),
        'humidity': random.uniform(30.0, 50.0),
        'timestamp': time.time()
    }
    return data

if __name__ == "__main__":
    while True:
        sensor_data = generate_sensor_data()
        try:
            producer.send('sensor_data', sensor_data)
            print(f"Sent: {sensor_data}")
        except Exception as e:
            print(f"Error sending data: {e}")
        time.sleep(1)