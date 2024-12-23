# tests/test_openmaint_consumer.py

import unittest
from unittest.mock import patch, MagicMock
from IntegrationWithExistingSystems import openmaint_consumer

class TestOpenMaintConsumer(unittest.TestCase):

    @patch('IntegrationWithExistingSystems.openmaint_consumer.load_config')
        mock_load_config.side_effect = [
        ]

        mock_consumer_instance = MagicMock()
        mock_message = MagicMock()

        mock_client_instance.get_work_order_details.return_value = {
        }



        mock_client_instance.create_work_order.assert_called_once()


if __name__ == '__main__':
    unittest.main()
