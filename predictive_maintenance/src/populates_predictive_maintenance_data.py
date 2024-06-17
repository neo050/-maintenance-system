import psycopg2
from datetime import datetime, timedelta
import random

# Database connection details
db_config = {
    'dbname': 'myprojectdb',
    'user': 'postgres',
    'password': 'neoray123',  # Replace with your actual password
    'host': 'localhost',
    'port': '5432'
}

# Connect to PostgreSQL
conn = psycopg2.connect(**db_config)
cur = conn.cursor()

# Add the timestamp column if it doesn't exist
cur.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='predictive_maintenance_data' AND column_name='timestamp') THEN
            ALTER TABLE predictive_maintenance_data ADD COLUMN timestamp TIMESTAMP;
        END IF;
    END $$;
""")
conn.commit()

# Define the start date for the timestamps
start_date = datetime(2024, 1, 1, 0, 0, 0)
time_increment = timedelta(minutes=1)

# Fetch the data to update
cur.execute("SELECT uid FROM predictive_maintenance_data ORDER BY uid")
rows = cur.fetchall()

# Update each row with a new timestamp
for index, row in enumerate(rows):
    timestamp = start_date + (index * time_increment)
    cur.execute("UPDATE predictive_maintenance_data SET timestamp = %s WHERE uid = %s", (timestamp, row[0]))

# Commit the changes and close the connection
conn.commit()
cur.close()
conn.close()

print("Timestamp column added and populated with data.")
