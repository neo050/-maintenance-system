# tests/test_sensor_data_simulator.py

import unittest
from unittest.mock import patch, MagicMock

class TestSensorDataSimulatorRunner(unittest.TestCase):

    @patch('RealTimeProcessing.SensorDataSimulator.SensorDataSimulator')
    def test_main(self, mock_simulator_class):
        mock_simulator_instance = MagicMock()
        mock_simulator_class.return_value = mock_simulator_instance

        # Run the main function

        # Check that the simulator was instantiated and started
        mock_simulator_class.assert_called_once()
        mock_simulator_instance.start_simulation.assert_called_once()
        mock_simulator_instance.cleanup.assert_called()

if __name__ == '__main__':
    unittest.main()
