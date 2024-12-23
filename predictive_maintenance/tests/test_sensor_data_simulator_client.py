import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
import subprocess
import socket
import os
import json
import sys

class TestSensorDataSimulator(unittest.TestCase):

    def setUp(self):
        self.simulator.logger = MagicMock()




    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
        self.simulator.create_producer()
            api_version=(3, 7, 0),
            value_serializer=unittest.mock.ANY
        )

    def test_simulate_sensor_data(self):
        df = self.simulator.simulate_sensor_data()
        self.assertFalse(df.empty)

    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')
        self.simulator.create_producer()

    @patch('RealTimeProcessing.SensorDataSimulatorClient.subprocess.run')
    @patch('RealTimeProcessing.SensorDataSimulatorClient.KafkaProducer')



        self.simulator.cleanup()

if __name__ == '__main__':
    unittest.main()
