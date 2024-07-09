# Kafka Setup Instructions

## Prerequisites

- Java JDK (version 8 or later)
- Apache Kafka (downloaded and extracted)

## Steps to Set Up Kafka

### 1. Move Kafka to a Shorter Path (Optional)

If your Kafka directory path is too long, you may encounter issues. You can move it to a shorter path:

1. Create a new directory:
   ```sh
   mkdir C:\kafka
2. Start ZooKeeper
Navigate to: cd C:\kafka\kafka_2.13-3.7.0\bin\windows
Start ZooKeeper: .\zookeeper-server-start.bat ..\..\config\zookeeper.properties

3. Start Kafka Broker
Open another command prompt window.

Navigate to the bin\windows directory in the Kafka directory:
cd C:\kafka\kafka_2.13-3.7.0\bin\windows

Start Kafka Broker:
.\kafka-server-start.bat ..\..\config\server.properties


4.  Create a Kafka Topic
Open another command prompt window.

Navigate to the bin\windows directory in the Kafka directory:
cd C:\kafka\kafka_2.13-3.7.0\bin\windows

Create a topic named sensor_topic:
.\kafka-topics.bat --create --topic sensor_topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

5. Firewall Configuration
When prompted by your firewall, allow access for Private networks to enable Kafka and ZooKeeper to communicate properly within your local network. Deny access for Public networks for security reasons.
