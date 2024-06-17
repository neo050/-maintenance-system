import psycopg2
import pandas as pd

# Database connection
conn = psycopg2.connect(
    dbname="myprojectdb",
    user="postgres",
    password="neoray123",
    host="localhost",
    port="5432"
)

# Create a cursor object
cur = conn.cursor()

# Load CSV data
csv_file_path = '../data/raw/sensor_data.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file_path)

# Print column names to verify
print("Column names in CSV:", data.columns.tolist())

# Insert CSV data into the database with conversion for boolean columns
for index, row in data.iterrows():
    cur.execute(
        """
        INSERT INTO predictive_maintenance_data (
            product_id, type, air_temperature, process_temperature, 
            rotational_speed, torque, tool_wear, 
            twf, hdf, pwf, osf, rnf, machine_failure
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            row['Product ID'], row['Type'], row['Air temperature [K]'],
            row['Process temperature [K]'], row['Rotational speed [rpm]'],
            row['Torque [Nm]'], row['Tool wear [min]'],
            bool(row['TWF']), bool(row['HDF']), bool(row['PWF']), bool(row['OSF']), bool(row['RNF']),
            bool(row['Machine failure'])
        )
    )

# Commit the transaction and close the connection
conn.commit()
cur.close()
conn.close()

print("Real data from CSV file populated in the database.")