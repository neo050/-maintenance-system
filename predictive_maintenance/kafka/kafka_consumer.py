from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'alert_topic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def consume_alerts():
    for message in consumer:
        print(message.value)

if __name__ == "__main__":
    consume_alerts()
