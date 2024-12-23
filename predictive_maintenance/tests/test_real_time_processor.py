# tests/test_real_time_processor.py
import unittest
from unittest.mock import patch, MagicMock

class TestRealTimeProcessorRunner(unittest.TestCase):
    @patch('RealTimeProcessing.RealTimeProcessor.RealTimeProcessor')
    def test_main(self, mock_processor_class):
        mock_processor_instance = MagicMock()
        mock_processor_class.return_value = mock_processor_instance

        # Run the main function

        # Check that the processor was instantiated and started
        mock_processor_class.assert_called_once()
        mock_processor_instance.process_messages.assert_called_once()

if __name__ == '__main__':
    unittest.main()
