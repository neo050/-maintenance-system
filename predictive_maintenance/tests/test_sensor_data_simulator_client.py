# tests/test_sensor_data_simulator_client.py

import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
import subprocess
import socket
import os
import json
import sys

from RealTimeProcessing.SensorDataSimulatorClient import SensorDataSimulator

class TestSensorDataSimulator(unittest.TestCase):

    def setUp(self):
        # Initialize the simulator
        self.simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])
        self.simulator.logger = MagicMock()

    def tearDown(self):
        # Ensure cleanup is called after each test
        self.simulator.cleanup()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run')
    @patch('RealTimeProcessing.SensorDataSimulatorClient.time.sleep')
    def test_init_docker_compose(self, mock_sleep, mock_subprocess_run):
        """
        Test that the constructor attempts to run Docker Compose
        but we mock it so it doesn't actually happen.
        """
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        mock_sleep.return_value = None

        # Re-instantiate the simulator (so Docker compose is called).
        simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])
        simulator.logger = MagicMock()

        mock_subprocess_run.assert_called_once()
        mock_sleep.assert_any_call(10)  # We skip real waiting

    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
    def test_create_producer(self, mock_producer_class):
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer
        self.simulator.create_producer()
        mock_producer_class.assert_called_once_with(
            bootstrap_servers=['localhost:9092'],
            api_version=(3, 7, 0),
            value_serializer=unittest.mock.ANY
        )
        self.assertIsNotNone(self.simulator.producer)

    def test_simulate_sensor_data(self):
        df = self.simulator.simulate_sensor_data()
        self.assertFalse(df.empty)
        self.assertIn('Air temperature [K]', df.columns)

    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
    def test_send_data(self, mock_producer_class):
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer
        self.simulator.create_producer()

        data = {'example': 'test'}
        self.simulator.send_data('test-topic', data)
        mock_producer.send.assert_called_once_with('test-topic', value=data)
        mock_producer.flush.assert_called_once()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.socket.create_connection')
    def test_is_kafka_running(self, mock_create_conn):
        mock_sock = MagicMock()
        mock_create_conn.return_value = mock_sock
        # The 'with' statement calls mock_sock.__enter__()
        mock_sock.__enter__.return_value = mock_sock

        # Now the code sees it as a valid context manager
        self.assertTrue(self.simulator.is_kafka_running())

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run')
    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
    def test_start_simulation(self, mock_kafka_producer, mock_subprocess_run):
        """
        Ensure no real Docker or Kafka calls, but the logic is tested.
        """
        mock_kafka_producer.return_value = MagicMock()
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.simulator.is_kafka_running = MagicMock(return_value=True)
        self.simulator.create_kafka_topic = MagicMock()
        self.simulator.num_samples = 2  # Fewer samples for speed
        self.simulator.start_simulation()

        self.simulator.is_kafka_running.assert_called_once()
        self.simulator.create_kafka_topic.assert_any_call('sensor-data')
        self.simulator.create_kafka_topic.assert_any_call('failure_predictions')

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run')
    def test_create_kafka_topic_failure(self, mock_subprocess_run):
        # If docker exec fails
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd='docker exec',
            output=b"",  # or some bytes
            stderr=b"Topic creation error"
        )
        with self.assertRaises(Exception) as ctx:
            self.simulator.create_kafka_topic('fail-topic')
        self.assertIn('Failed to create Kafka topic:', str(ctx.exception))

    def test_cleanup(self):
        # Just ensure it doesn't blow up
        self.simulator.cleanup()
        self.simulator.logger.info.assert_called_with("Kafka producer has been shut down.")

if __name__ == '__main__':
    unittest.main()
