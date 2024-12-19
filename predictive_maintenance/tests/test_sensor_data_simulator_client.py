import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
from RealTimeProcessing.SensorDataSimulatorClient import SensorDataSimulator
import subprocess
import socket
import os
import json
import sys

class TestSensorDataSimulator(unittest.TestCase):

    def setUp(self):
        current_dir = r"C:\Users\neora\Desktop\Final_project\-maintenance-system\predictive_maintenance\RealTimeProcessing"
        kafka_dir = os.path.abspath(os.path.join(current_dir, 'kafka', 'kafka_2.13-3.7.0'))
        self.kafka_dir = kafka_dir
        self.simulator = SensorDataSimulator(kafka_dir=self.kafka_dir)
        # Mock the logger to suppress logging during tests
        self.simulator.logger = MagicMock()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.psutil.process_iter')
    def test_terminate_processes(self, mock_process_iter):
        mock_proc = Mock()
        mock_proc.info = {
            'pid': 1234,
            'name': 'fake_process',
            'cmdline': ['fake_process']
        }
        mock_process_iter.return_value = [mock_proc]

        with patch.object(mock_proc, 'terminate') as mock_terminate, \
                patch.object(mock_proc, 'wait') as mock_wait:
            self.simulator.terminate_processes(['fake_process'])
            mock_terminate.assert_called_once()
            mock_wait.assert_called_once_with(timeout=30)

    @patch('RealTimeProcessing.SensorDataSimulatorClient.shutil.rmtree')
    @patch('RealTimeProcessing.SensorDataSimulatorClient.os.path.exists')
    def test_delete_log_directories(self, mock_exists, mock_rmtree):
        mock_exists.return_value = True
        log_dirs = ['/fake/log/dir1', '/fake/log/dir2']
        self.simulator.delete_log_directories(log_dirs)
        self.assertEqual(mock_rmtree.call_count, 2)

    @patch('RealTimeProcessing.SensorDataSimulatorClient.os.path.exists')
    @patch('RealTimeProcessing.SensorDataSimulatorClient.os.makedirs')
    def test_ensure_directory_exists(self, mock_makedirs, mock_exists):
        mock_exists.return_value = False
        directory = '/fake/directory'
        self.simulator.ensure_directory_exists(directory)
        mock_makedirs.assert_called_once_with(directory)

        mock_exists.return_value = True
        mock_makedirs.reset_mock()
        self.simulator.ensure_directory_exists(directory)
        mock_makedirs.assert_not_called()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.socket.socket')
    def test_is_port_in_use(self, mock_socket_class):
        mock_socket = Mock()
        mock_socket_class.return_value.__enter__.return_value = mock_socket

        # Port is in use
        mock_socket.connect_ex.return_value = 0
        self.assertTrue(self.simulator.is_port_in_use(9092))

        # Port is not in use
        mock_socket.connect_ex.return_value = 1
        self.assertFalse(self.simulator.is_port_in_use(9092))

    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
    def test_create_producer(self, mock_kafka_producer):
        mock_kafka_producer.return_value = Mock()
        self.simulator.create_producer()
        mock_kafka_producer.assert_called_once_with(
            bootstrap_servers=self.simulator.bootstrap_servers,
            api_version=(3, 7, 0),
            value_serializer=unittest.mock.ANY
        )

    def test_simulate_sensor_data(self):
        df = self.simulator.simulate_sensor_data()
        self.assertFalse(df.empty)
        expected_columns = [
            'UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure',
            'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
        ]
        self.assertTrue(all(col in df.columns for col in expected_columns))
        self.assertTrue(df['Air temperature [K]'].dtype == np.float64)
        self.assertTrue(df['Type'].dtype == object)

    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
    def test_send_data(self, mock_kafka_producer):
        mock_producer_instance = mock_kafka_producer.return_value
        self.simulator.producer = mock_producer_instance
        data = {'key': 'value'}
        self.simulator.send_data('test-topic', data)
        mock_producer_instance.send.assert_called_once_with('test-topic', value=data)
        mock_producer_instance.flush.assert_called_once()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.Popen')
    def test_create_kafka_topic(self, mock_popen):
        with patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            self.simulator.create_kafka_topic('test-topic')
            mock_run.assert_called()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.Popen')
    def test_start_zookeeper(self, mock_popen):
        with patch('RealTimeProcessing.SensorDataSimulatorClient.SensorDataSimulator.is_port_in_use', return_value=False):
            result = self.simulator.start_zookeeper()
            mock_popen.assert_called()
            self.assertTrue(result)

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.Popen')
    def test_start_kafka_broker(self, mock_popen):
        with patch('RealTimeProcessing.SensorDataSimulatorClient.SensorDataSimulator.wait_for_kafka_ready', return_value=True), \
             patch('RealTimeProcessing.SensorDataSimulatorClient.os.path.isfile', return_value=True):
            result = self.simulator.start_kafka_broker()
            mock_popen.assert_called()
            self.assertTrue(result)

    @patch('RealTimeProcessing.SensorDataSimulatorClient.time.sleep')
    @patch('RealTimeProcessing.SensorDataSimulatorClient.SensorDataSimulator.send_data')
    def test_start_simulation(self, mock_send_data, mock_sleep):
        self.simulator.simulate_sensor_data = MagicMock(return_value=pd.DataFrame({
            'Air temperature [K]': [300.0],
            'Process temperature [K]': [310.0],
            'Rotational speed [rpm]': [1500.0],
            'Torque [Nm]': [40.0],
            'Tool wear [min]': [100],
            'Type': ['L'],
            'UDI': [1],
            'Product ID': ['L12345'],
            'Machine failure': [0],
            'TWF': [0],
            'HDF': [0],
            'PWF': [0],
            'OSF': [0],
            'RNF': [0]
        }))
        with patch.object(self.simulator, 'start_zookeeper', return_value=True), \
                patch.object(self.simulator, 'start_kafka_broker', return_value=True), \
                patch.object(self.simulator, 'create_kafka_topic'), \
                patch.object(self.simulator, 'create_producer'):
            self.simulator.start_simulation()
            mock_send_data.assert_called()
            mock_sleep.assert_called()

    def test_error_handling_when_port_in_use(self):
        with patch('RealTimeProcessing.SensorDataSimulatorClient.SensorDataSimulator.is_port_in_use', return_value=True), \
             patch('RealTimeProcessing.SensorDataSimulatorClient.SensorDataSimulator.wait_for_zookeeper_ready', return_value=False):
            with self.assertRaises(Exception) as ctx:
                self.simulator.start_zookeeper()
            self.assertIn("Zookeeper is not ready.", str(ctx.exception))

    @patch('RealTimeProcessing.SensorDataSimulatorClient.os.path.isfile', return_value=False)
    @patch('RealTimeProcessing.SensorDataSimulatorClient.SensorDataSimulator.is_port_in_use', return_value=False)
    def test_start_zookeeper_command_missing(self, mock_port_in_use, mock_isfile):
        # Now is_port_in_use returns False, and isfile returns False
        # This should cause FileNotFoundError to be raised.
        with self.assertRaises(FileNotFoundError):
            self.simulator.start_zookeeper()
        self.simulator.logger.error.assert_called()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.SensorDataSimulator.is_port_in_use', return_value=False)
    @patch('RealTimeProcessing.SensorDataSimulatorClient.socket.create_connection', side_effect=ConnectionRefusedError)
    def test_start_zookeeper_never_ready(self, mock_socket, mock_port_check):
        # Expect the logger to be called with "Zookeeper did not become ready in time."
        # and the exception to contain "Zookeeper did not start in time."
        with self.assertRaises(Exception) as context:
            self.simulator.start_zookeeper()
        self.assertIn("Zookeeper did not start in time.", str(context.exception))
        self.simulator.logger.error.assert_called_with("Zookeeper did not become ready in time.")

    def test_simulate_sensor_data_zero_samples(self):
        self.simulator.num_samples = 0
        df = self.simulator.simulate_sensor_data()
        self.assertTrue(df.empty)

    def test_simulate_sensor_data_large_samples(self):
        self.simulator.num_samples = 10000
        df = self.simulator.simulate_sensor_data()
        self.assertEqual(len(df), 10000)

    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer', side_effect=Exception("Producer failure"))
    def test_create_producer_failure(self, mock_kafka_producer):
        with self.assertRaises(Exception) as context:
            self.simulator.create_producer()
        self.assertIn("Failed to create Kafka producer:", str(context.exception))
        self.simulator.logger.error.assert_called()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run')
    def test_create_kafka_topic_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="Permission denied")
        with self.assertRaises(Exception) as context:
            self.simulator.create_kafka_topic('test-fail-topic')
        self.assertIn("Failed to create Kafka topic:", str(context.exception))
        self.simulator.logger.error.assert_called()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
    def test_send_data_failure(self, mock_kafka_producer):
        mock_producer_instance = mock_kafka_producer.return_value
        mock_producer_instance.send.side_effect = Exception("Send failure")
        self.simulator.producer = mock_producer_instance

        data = {'key': 'value'}
        self.simulator.send_data('test-topic', data)
        self.simulator.logger.error.assert_called_with('Failed to send data: Send failure')

    def test_cleanup_without_start(self):
        self.simulator.cleanup()
        self.simulator.logger.info.assert_any_call("Shutting down Kafka broker and Zookeeper...")

    def test_cleanup_twice(self):
        self.simulator.cleanup()
        self.simulator.cleanup()
        self.simulator.logger.info.assert_any_call("Shutting down Kafka broker and Zookeeper...")

if __name__ == '__main__':
    unittest.main()
