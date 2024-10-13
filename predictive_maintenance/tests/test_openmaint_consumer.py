# tests/test_openmaint_consumer.py

import unittest
from unittest.mock import patch, MagicMock
from IntegrationWithExistingSystems import openmaint_consumer
from IntegrationWithExistingSystems.OpenMaintClient import OpenMaintClient


class TestOpenMaintConsumer(unittest.TestCase):

    @patch('IntegrationWithExistingSystems.openmaint_consumer.managed_openmaint_client')
    @patch('IntegrationWithExistingSystems.openmaint_consumer.create_kafka_consumer')
    @patch('IntegrationWithExistingSystems.openmaint_consumer.load_config')
    def test_openmaint_consumer_main(
        self,
        mock_load_config,
        mock_create_consumer,
        mock_managed_client
    ):
        """
        Test the openmaint_consumer_main logic,
        ensuring no real Docker or HTTP calls are made,
        and that messages are processed as expected.
        """
        # 1) Mock configurations
        mock_load_config.side_effect = [
            # For kafka_config.yaml
            {'kafka': {
                'bootstrap_servers': ['127.0.0.1:9092'],
                'failure_predictions_topic': 'failure_predictions'
            }},
            # For openmaint_config.yaml
            {'openmaint': {
                'api_url': 'http://fake-api-url',
                'username': 'user',
                'password': 'pass'
            }}
        ]

        # 2) Mock Kafka consumer
        mock_consumer_instance = MagicMock()
        # Simulate just one message
        mock_message = MagicMock()
        mock_message.value = {
            'PredictedFailureDate': '2024-12-20 10:00:00',
            'RiskScore': 0.9
        }
        mock_consumer_instance.poll.return_value = {
            None: [mock_message]
        }
        mock_create_consumer.return_value = mock_consumer_instance

        # 3) Mock the managed_openmaint_client context manager
        mock_client_instance = MagicMock(spec=OpenMaintClient)
        # For create_work_order, return some ID
        mock_client_instance.create_work_order.return_value = 'wo-123'
        # For get_work_order_details, return a dict
        mock_client_instance.get_work_order_details.return_value = {
            '_id': 'wo-123',
            'some': 'details'
        }
        mock_managed_client.return_value.__enter__.return_value = mock_client_instance

        # 4) Call openmaint_consumer_main with a mocked shutdown_event
        shutdown_event = MagicMock()
        shutdown_event.is_set.side_effect = [False, True]  # Process 1 iteration, then stop

        openmaint_consumer.openmaint_consumer_main(shutdown_event)

        # 5) Assertions
        mock_load_config.assert_called()
        mock_create_consumer.assert_called_once()

        # Ensure create_work_order was called
        mock_client_instance.create_work_order.assert_called_once()
        # Also ensure get_work_order_details was called
        mock_client_instance.get_work_order_details.assert_called_once()

        # The message was consumed and processed
        self.assertTrue(mock_consumer_instance.poll.called)


if __name__ == '__main__':
    unittest.main()
