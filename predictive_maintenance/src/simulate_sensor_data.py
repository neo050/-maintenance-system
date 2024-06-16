import psycopg2
from datetime import datetime
import random

# Connect to PostgreSQL database without SSL
conn = psycopg2.connect(
    dbname="myprojectdb",
    user="postgres",
    password="your_password",
    host="localhost",
    port="5432",
    sslmode='disable'  # Disable SSL
)

# Create a cursor object
cur = conn.cursor()

# Insert simulated data
for _ in range(100):
    timestamp = datetime.now()
    sensor_id = random.randint(1, 100)
    temperature = random.uniform(20.0, 25.0)
    pressure = random.uniform(1.0, 2.0)
    humidity = random.uniform(30.0, 50.0)
    cur.execute(
        "INSERT INTO equipment_data (sensor_id, temperature, pressure, humidity, timestamp) VALUES (%s, %s, %s, %s, %s)",
        (sensor_id, temperature, pressure, humidity, timestamp)
    )

# Commit the transaction and close the connection
conn.commit()
cur.close()
conn.close()
