# tests/test_openmaint_consumer.py

import unittest
from unittest.mock import patch, MagicMock
from IntegrationWithExistingSystems import openmaint_consumer

class TestOpenMaintConsumer(unittest.TestCase):

    @patch('IntegrationWithExistingSystems.openmaint_consumer.OpenMaintClient')
    @patch('IntegrationWithExistingSystems.openmaint_consumer.KafkaConsumer')
    @patch('IntegrationWithExistingSystems.openmaint_consumer.load_config')
    def test_main(self, mock_load_config, mock_kafka_consumer, mock_openmaint_client):
        # Mock configurations
        mock_load_config.side_effect = [
            {'kafka': {'bootstrap_servers': ['127.0.0.1:9092']}},
            {'openmaint': {'api_url': 'http://fake-api-url', 'username': 'user', 'password': 'pass'}}
        ]

        # Mock Kafka consumer
        mock_consumer_instance = MagicMock()
        mock_message = MagicMock()
        mock_message.value = {'PredictedFailureDate': '2021-01-01 00:00:00', 'RiskScore': 0.9}
        mock_consumer_instance.__iter__.return_value = [mock_message]
        mock_kafka_consumer.return_value = mock_consumer_instance

        # Mock OpenMaintClient instance
        mock_client_instance = MagicMock()
        # Ensure create_work_order returns a string (simulating a real work order ID)
        mock_client_instance.create_work_order.return_value = 'work_order_id'
        # Set get_work_order_details to return a dictionary (serializable)
        mock_client_instance.get_work_order_details.return_value = {
            '_id': 'work_order_id',
            'details': 'some details'
        }

        mock_openmaint_client.return_value = mock_client_instance

        # Run main function
        openmaint_consumer.main()

        # Assertions
        mock_kafka_consumer.assert_called_once()
        mock_openmaint_client.assert_called_once_with('http://fake-api-url', 'user', 'pass')
        mock_client_instance.create_work_order.assert_called_once()
        # Also check that get_work_order_details was called once with 'work_order_id'
        mock_client_instance.get_work_order_details.assert_called_once_with('work_order_id')


if __name__ == '__main__':
    unittest.main()
