# tests/test_real_time_processor_client.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from RealTimeProcessing.RealTimeProcessorClient import RealTimeProcessor

class TestRealTimeProcessor(unittest.TestCase):

    def setUp(self):
        self.models_dir = '../models'
        self.config_file = '../config/database_config.yaml'
        self.processor = RealTimeProcessor(self.models_dir, self.config_file)
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
        }
        df = pd.DataFrame(sample_data)
        df_fe = self.processor.feature_engineering(df)
        self.assertIsNone(df_fe)  # Should return None due to missing 'Type'

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
        mock_message = MagicMock()
        mock_message.value = {
            'Process temperature [K]': 310.0,
            'Rotational speed [rpm]': 1500.0,
            'Torque [Nm]': 40.0,
            'Tool wear [min]': 100,
            'Type': 'L'
        }
        self.processor.process_messages()
        self.processor.logger.error.assert_called()

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
        self.processor.setup_kafka_consumer()
        self.assertIsNotNone(self.processor.consumer)

    @patch('RealTimeProcessing.RealTimeProcessorClient.KafkaProducer')
    def test_setup_kafka_producer(self, mock_kafka_producer):
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

    @patch('RealTimeProcessing.RealTimeProcessorClient.RealTimeProcessor.save_to_database')
    @patch('RealTimeProcessing.RealTimeProcessorClient.RealTimeProcessor.prepare_data')
    @patch('RealTimeProcessing.RealTimeProcessorClient.RealTimeProcessor.feature_engineering')


        mock_message = MagicMock()
        mock_message.value = {
            'Air temperature [K]': 300.0,
            'Process temperature [K]': 310.0,
            'Rotational speed [rpm]': 1500.0,
            'Torque [Nm]': 40.0,
            'Tool wear [min]': 100,
            'Type': 'L'
        }

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

        self.processor.models = {
        }
            scaler = self.processor.models[model_type]['scaler']
            scaler.transform.return_value = df[scaler.feature_names_in_].values
                if model_type == 'supervised':
                else:

            ]

        self.processor.process_messages()



if __name__ == '__main__':
    unittest.main()
