# tests/test_real_time_processor_client.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from RealTimeProcessing.RealTimeProcessorClient import RealTimeProcessor


class TestRealTimeProcessor(unittest.TestCase):

    @patch('RealTimeProcessing.RealTimeProcessorClient.logging.getLogger')
    def setUp(self, mock_get_logger):
        self.models_dir = '../models'
        self.config_file = '../config/database_config.yaml'
        self.mock_logger = MagicMock()
        mock_get_logger.return_value = self.mock_logger

        self.processor = RealTimeProcessor(self.models_dir, self.config_file)

    def tearDown(self):
        # Ensure cleanup is called after each test
        self.processor.cleanup()

    def test_aggregate_predictions(self):
        predictions = [0.2, 0.6, 0.8]
        final_pred, final_score = self.processor.aggregate_predictions(predictions)
        self.assertEqual(final_pred, 1)
        self.assertAlmostEqual(final_score, 0.5333, places=4)

    def test_feature_engineering_with_missing_type(self):
        sample_data = {
            'Air temperature [K]': [300.0],
            'Process temperature [K]': [310.0],
            'Rotational speed [rpm]': [1500.0],
            'Torque [Nm]': [40.0],
            'Tool wear [min]': [100]
            # 'Type' column is missing
        }
        df = pd.DataFrame(sample_data)
        df_fe = self.processor.feature_engineering(df)
        self.assertIsNone(df_fe)  # Should return None due to missing 'Type'

        # Assert that an error was logged for missing 'Type'
        self.mock_logger.error.assert_called_with("Type column is missing from the input data.")

    @patch('RealTimeProcessing.RealTimeProcessorClient.pd.DataFrame.to_sql')
    def test_save_to_database(self, mock_to_sql):
        self.processor.engine = MagicMock()
        df = pd.DataFrame({
            'Air temperature [K]': [300.0],
            'final_prediction': [1],
            'final_score': [0.7],
            'timestamp': [pd.Timestamp.now()]
        })
        self.processor.save_to_database(df, 'test_table')
        mock_to_sql.assert_called_once()

    @patch('RealTimeProcessing.RealTimeProcessorClient.KafkaConsumer')
    @patch('RealTimeProcessing.RealTimeProcessorClient.KafkaProducer')
    def test_process_messages_with_invalid_data(self, mock_kafka_producer, mock_kafka_consumer):
        # Mock Kafka consumer with invalid data
        mock_consumer = MagicMock()
        mock_message = MagicMock()
        mock_message.value = {
            'Air temperature [K]': 'invalid',  # Invalid type
            'Process temperature [K]': 310.0,
            'Rotational speed [rpm]': 1500.0,
            'Torque [Nm]': 40.0,
            'Tool wear [min]': 100,
            'Type': 'L'
        }
        mock_consumer.__iter__.return_value = [mock_message]
        mock_kafka_consumer.return_value = mock_consumer

        mock_producer = MagicMock()
        mock_kafka_producer.return_value = mock_producer

        self.processor.process_messages()
        # An error should be logged due to invalid 'Air temperature [K]'
        self.mock_logger.error.assert_called_with(
            "Invalid data type in column 'Air temperature [K]'. Expected types: (<class 'int'>, <class 'float'>)")

    @patch('RealTimeProcessing.RealTimeProcessorClient.os.listdir')
    @patch('RealTimeProcessing.RealTimeProcessorClient.load_model')
    @patch('RealTimeProcessing.RealTimeProcessorClient.joblib.load')
    def test_load_models(self, mock_joblib_load, mock_load_model, mock_listdir):
        mock_listdir.return_value = ['model_fold_1.keras', 'scaler_nn.pkl']
        mock_joblib_load.return_value = MagicMock()
        mock_load_model.return_value = MagicMock()

        self.processor.load_models()
        self.assertTrue(len(self.processor.models['cnn']['models']) > 0)
        self.assertIsNotNone(self.processor.models['cnn']['scaler'])

    @patch('builtins.open')
    @patch('RealTimeProcessing.RealTimeProcessorClient.yaml.safe_load')
    def test_load_config(self, mock_yaml_load, mock_open):
        mock_yaml_load.return_value = {'database': {'url': 'postgresql://user:pass@localhost/db'}}
        self.processor.load_config()
        self.assertIsNotNone(self.processor.config)
        self.assertIn('database', self.processor.config)

    @patch('RealTimeProcessing.RealTimeProcessorClient.create_engine')
    def test_create_database_engine(self, mock_create_engine):
        self.processor.config = {'database': {'url': 'postgresql://user:pass@localhost/db'}}
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        self.processor.create_database_engine()
        mock_create_engine.assert_called_once_with('postgresql://user:pass@localhost/db')
        self.assertIsNotNone(self.processor.engine)

    @patch('RealTimeProcessing.RealTimeProcessorClient.KafkaConsumer')
    def test_setup_kafka_consumer(self, mock_kafka_consumer):
        mock_consumer_instance = MagicMock()
        mock_kafka_consumer.return_value = mock_consumer_instance
        self.processor.setup_kafka_consumer()
        self.assertIsNotNone(self.processor.consumer)

    @patch('RealTimeProcessing.RealTimeProcessorClient.KafkaProducer')
    def test_setup_kafka_producer(self, mock_kafka_producer):
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance
        self.processor.setup_kafka_producer()
        self.assertIsNotNone(self.processor.producer)

    def test_feature_engineering(self):
        sample_data = {
            'Air temperature [K]': [300.0],
            'Process temperature [K]': [310.0],
            'Rotational speed [rpm]': [1500.0],
            'Torque [Nm]': [40.0],
            'Tool wear [min]': [100],
            'Type': ['L']
        }
        df = pd.DataFrame(sample_data)
        df_fe = self.processor.feature_engineering(df)
        self.assertIn('OSF_threshold', df_fe.columns)

    def test_prepare_data(self):
        df = pd.DataFrame({
            'Air temperature [K]': [300.0],
            'Process temperature [K]': [310.0],
            'Rotational speed [rpm]': [1500.0],
            'Torque [Nm]': [40.0],
            'Tool wear [min]': [100],
            'Type': [1],
            'Temp_diff': [10.0],
            'Rotational speed [rad/s]': [157.08],
            'Power': [6283.2],
            'Tool_Torque_Product': [4000.0],
            'TWF_condition': [0],
            'HDF_condition': [0],
            'PWF_condition': [0],
            'OSF_condition': [0],
            'Failure_Risk': [0]
        })
        df_prepared = self.processor.prepare_data(df)
        self.assertIsNotNone(df_prepared)

    @patch.object(RealTimeProcessor, 'setup_kafka_consumer')
    @patch.object(RealTimeProcessor, 'setup_kafka_producer')
    @patch.object(RealTimeProcessor, 'load_models')  # We don't want it to load real models
    @patch('RealTimeProcessing.RealTimeProcessorClient.RealTimeProcessor.save_to_database')
    @patch('RealTimeProcessing.RealTimeProcessorClient.RealTimeProcessor.prepare_data')
    @patch('RealTimeProcessing.RealTimeProcessorClient.RealTimeProcessor.feature_engineering')
    def test_process_messages(
            self,
            mock_feature_engineering,
            mock_prepare_data,
            mock_save_to_db,
            mock_load_models,
            mock_setup_producer,
            mock_setup_consumer
    ):
        """
        This test ensures we never attempt real Kafka connections,
        but do run through the process_messages logic with one mock message.
        """

        # 1) Mock out 'setup_kafka_consumer()' so it never calls the real code
        mock_setup_consumer.return_value = None

        # 2) Mock out 'setup_kafka_producer()'
        mock_setup_producer.return_value = None

        # 3) Mock 'load_models()' so we don't load real models
        mock_load_models.return_value = None

        # 4) Manually set up a mock consumer that yields exactly 1 message
        mock_consumer = MagicMock()
        mock_message = MagicMock()
        mock_message.value = {
            'Air temperature [K]': 300.0,
            'Process temperature [K]': 310.0,
            'Rotational speed [rpm]': 1500.0,
            'Torque [Nm]': 40.0,
            'Tool wear [min]': 100,
            'Type': 'L'
        }
        mock_consumer.__iter__.return_value = [mock_message]

        # 5) Also mock the producer attribute
        mock_producer = MagicMock()
        self.processor.consumer = mock_consumer
        self.processor.producer = mock_producer

        # 6) Provide mock data frames
        df = pd.DataFrame([{
            'Air temperature [K]': 300.0,
            'Process temperature [K]': 310.0,
            'Rotational speed [rpm]': 1500.0,
            'Torque [Nm]': 40.0,
            'Tool wear [min]': 100,
            'Type': 'L',
            'Temp_diff': 10.0,
            'Rotational speed [rad/s]': 157.08,
            'Power': 6283.2,
            'Tool_Torque_Product': 4000.0,
            'TWF_condition': 0,
            'HDF_condition': 0,
            'PWF_condition': 0,
            'OSF_condition': 0,
            'Failure_Risk': 0
        }])
        mock_feature_engineering.return_value = df
        mock_prepare_data.return_value = df

        # 7) Mock the self.processor.models with fake scalers + models
        self.processor.models = {
            'supervised': {'models': [MagicMock()], 'scaler': MagicMock(feature_names_in_=df.columns)},
            'cnn': {'models': [MagicMock()], 'scaler': MagicMock(feature_names_in_=df.columns)},
            'lstm': {'models': [MagicMock()], 'scaler': MagicMock(feature_names_in_=df.columns)},
            'cnn_lstm': {'models': [MagicMock()], 'scaler': MagicMock(feature_names_in_=df.columns)},
        }
        # For each model, mock the transform & predict
        for model_type in self.processor.models:
            scaler = self.processor.models[model_type]['scaler']
            scaler.transform.return_value = df[self.processor.models[model_type]['scaler'].feature_names_in_].values
            for m in self.processor.models[model_type]['models']:
                if model_type == 'supervised':
                    m.predict.return_value = [0]
                else:
                    m.predict.return_value = np.array([[0.0]])

        # 8) Fill the buffers so that CNN, LSTM, CNN_LSTM won't skip
        for mtype in ['cnn', 'lstm', 'cnn_lstm']:
            self.processor.nn_sequence_buffers[mtype] = [
                df.values[0] for _ in range(self.processor.sequence_length - 1)
            ]

        # 9) Finally call process_messages
        self.processor.process_messages()

        # 10) Now verify our mocks were called
        mock_feature_engineering.assert_called_once()
        mock_prepare_data.assert_called_once()
        mock_save_to_db.assert_called()
        # Also verify aggregator, or final predictions, etc.

        # Optionally, verify that models were used for prediction
        for model_type in ['supervised', 'cnn', 'lstm', 'cnn_lstm']:
            for model in self.processor.models[model_type]['models']:
                model.predict.assert_called()

        # Verify that prediction was sent to Kafka if final_score >= 0.6
        # Since final_score is mocked as 0.0 or 0, it won't send to Kafka
        # Adjust mocks accordingly if you want to test Kafka sending


if __name__ == '__main__':
    unittest.main()
