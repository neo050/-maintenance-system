import json
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from keras.models import load_model
import pickle

# Load trained models
lstm_model = load_model('../models/lstm_model.keras')
with open('../models/isolation_forest_model.pkl', 'rb') as f:
    isolation_forest_model = pickle.load(f)

# Set up Kafka consumer and producer
consumer = KafkaConsumer(
    'sensor_topic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

print("Kafka Consumer and Producer are set up and running.")

# Process incoming messages
for message in consumer:
    data = message.value
    print(f"Received data: {data}")

    # Convert incoming data to DataFrame
    df = pd.DataFrame([data])

    # Preprocess data: scale features
    scaled_data = df.copy()
    for column in df.columns:
        scaled_data[column] = (df[column] - df[column].mean()) / df[column].std()

    # Predict with LSTM model
    lstm_prediction = lstm_model.predict(scaled_data)
    print(f"LSTM prediction: {lstm_prediction}")

    # Detect anomaly with Isolation Forest
    anomaly_prediction = isolation_forest_model.predict(scaled_data)
    print(f"Anomaly prediction: {anomaly_prediction}")

    # Send anomaly detection result to another topic if anomaly is detected
    if anomaly_prediction[0] == -1:
        alert_message = {
            'alert': 'Anomaly detected',
            'data': data
        }
        producer.send('alert_topic', value=alert_message)
        print(f"Sent alert: {alert_message}")
