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

    def tearDown(self):
        # Ensure cleanup is called after each test if simulator exists
        if hasattr(self, 'simulator') and self.simulator is not None:
            try:
                self.simulator.cleanup()
            except Exception:
                pass  # Suppress any exceptions during cleanup

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run')
    @patch('RealTimeProcessing.SensorDataSimulatorClient.time.sleep')
    def test_init_docker_compose(self, mock_sleep, mock_subprocess_run):
        """
        Test that the constructor attempts to run Docker Compose
        but we mock it so it doesn't actually happen.
        """
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        mock_sleep.return_value = None

        # Define the expected docker-compose path
        docker_compose_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                                           'docker-compose.yml')

        # Use assertLogs to capture log messages during initialization
        with self.assertLogs('RealTimeProcessing.SensorDataSimulatorClient', level='INFO') as cm:
            self.simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])

        # Assert that 'Kafka cluster started successfully.' was logged
        expected_message = "INFO:RealTimeProcessing.SensorDataSimulatorClient:Kafka cluster started successfully."
        self.assertIn(expected_message, cm.output)

        # Verify that 'subprocess.run' was called correctly
        mock_subprocess_run.assert_called_once_with(['docker-compose', '-f', docker_compose_path, 'up', '-d'],
                                                    check=True)
        mock_sleep.assert_any_call(10)  # We skip real waiting

    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
    def test_create_producer(self, mock_producer_class):
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer
        self.simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])
        self.simulator.create_producer()
        mock_producer_class.assert_called_once_with(
            bootstrap_servers=['localhost:9092'],
            api_version=(3, 7, 0),
            value_serializer=unittest.mock.ANY
        )
        self.assertIsNotNone(self.simulator.producer)

    def test_simulate_sensor_data(self):
        self.simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])
        df = self.simulator.simulate_sensor_data()
        self.assertFalse(df.empty)
        self.assertIn('Air temperature [K]', df.columns)

    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
    def test_send_data(self, mock_kafka_producer):
        mock_producer = MagicMock()
        mock_kafka_producer.return_value = mock_producer
        self.simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])
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
        self.simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])
        self.assertTrue(self.simulator.is_kafka_running())

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run')
    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
    def test_start_simulation(self, mock_kafka_producer, mock_subprocess_run):
        """
        Ensure no real Docker or Kafka calls, but the logic is tested.
        """
        mock_kafka_producer.return_value = MagicMock()
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])
        self.simulator.is_kafka_running = MagicMock(return_value=True)
        self.simulator.create_kafka_topic = MagicMock()
        self.simulator.num_samples = 2  # Fewer samples for speed
        self.simulator.start_simulation()

        self.simulator.is_kafka_running.assert_called_once()
        self.simulator.create_kafka_topic.assert_any_call('sensor-data')
        self.simulator.create_kafka_topic.assert_any_call('failure_predictions')

    def test_create_kafka_topic_failure(self):
        # Allow 'docker-compose up -d' to run successfully
        with patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run') as mock_run:
            # Define a side_effect function that handles different subprocess.run calls
            def side_effect_run(*args, **kwargs):
                cmd = args[0]
                if cmd[0] == 'docker-compose':
                    # Simulate successful 'docker-compose up -d'
                    return MagicMock(returncode=0, stdout=b"", stderr=b"")
                else:
                    # For any other command, return a default successful mock
                    return MagicMock(returncode=0, stdout=b"", stderr=b"")

            mock_run.side_effect = side_effect_run

            # Instantiate the simulator (this will call 'docker-compose up -d')
            self.simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])

        # Now, patch 'docker exec' for create_kafka_topic to simulate failure
        with patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run') as mock_run_exec:
            # Configure the mock to raise a CalledProcessError when 'docker exec' is called
            mock_run_exec.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd='docker exec',
                output=b"",
                stderr=b"Topic creation error"
            )

            # Attempt to create a Kafka topic and expect an exception
            with self.assertRaises(Exception) as ctx:
                self.simulator.create_kafka_topic('fail-topic')
            self.assertIn('Failed to create Kafka topic:', str(ctx.exception))

    def test_cleanup(self):
        self.simulator = SensorDataSimulator(bootstrap_servers=['localhost:9092'])
        self.simulator.producer = MagicMock()

        # Use assertLogs to capture log messages
        with self.assertLogs('RealTimeProcessing.SensorDataSimulatorClient', level='INFO') as cm:
            self.simulator.cleanup()

        # Define expected log messages
        expected_messages = [
            "INFO:RealTimeProcessing.SensorDataSimulatorClient:Closing the Kafka producer.",
            "INFO:RealTimeProcessing.SensorDataSimulatorClient:SensorDataSimulatorClient is shutting down."
        ]

        # Assert that both expected messages are in the log output
        for message in expected_messages:
            self.assertIn(message, cm.output)


if __name__ == '__main__':
    unittest.main()
