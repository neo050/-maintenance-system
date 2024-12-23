# RealTimeProcessing/RealTimeProcessorClient.py

import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from kafka import KafkaConsumer, KafkaProducer
from sqlalchemy import create_engine
import yaml
from datetime import datetime, timezone
import warnings
import tensorflow as tf
import time

# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


class RealTimeProcessor:
    def __init__(self, models_dir, config_file, bootstrap_servers=['localhost:9092'], log_file_path=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'logs','real_time_processing.log')):
        self.models_dir = models_dir
        self.config_file = config_file
        self.logger = self.setup_logger(log_file_path)
        self.models = {
            'cnn': {'models': [], 'scaler': None},
            'cnn_lstm': {'models': [], 'scaler': None},
            'lstm': {'models': [], 'scaler': None},
            'supervised': {'models': [], 'scaler': None},
        }
        self.type_mapping = {'H': 0, 'L': 1, 'M': 2}
        self.original_features = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Type'
        ]
        self.selected_features = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Temp_diff',
            'Rotational speed [rad/s]',
            'Power',
            'Tool_Torque_Product',
            'TWF_condition',
            'HDF_condition',
            'PWF_condition',
            'Failure_Risk',
            'Type',
            'OSF_condition'
        ]
        self.features_nn = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Temp_diff',
            'Rotational speed [rad/s]',
            'Power',
            'Tool_Torque_Product',
            'TWF_condition',
            'HDF_condition',
            'PWF_condition',
            'OSF_condition',
            'Failure_Risk'
        ]
        self.sequence_length = 5
        self.nn_sequence_buffers = {
            'cnn': [],
            'lstm': [],
            'cnn_lstm': [],
        }
        self.consumer = None
        self.producer = None
        self.engine = None
        self.config = None
        self.bootstrap_servers = bootstrap_servers

        self.load_models()
        self.load_config()
        self.create_database_engine()
        self.setup_kafka_consumer()
        self.setup_kafka_producer()

    def setup_logger(self, log_file_path):
        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Capture all logs

        # Create handlers
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # File handler for logging to a file (DEBUG level and above)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Stream handler for logging to the console (INFO level and above)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    def load_models(self):
        models_dir = self.models_dir
        models = self.models

        # CNN models
        cnn_model_dir = os.path.join(models_dir, 'cnn_model_combined')
        cnn_model_files = [
            os.path.join(cnn_model_dir, file)
            for file in os.listdir(cnn_model_dir) if file.endswith('.keras')
        ]
        # Load scaler for CNN models
        scaler_file = os.path.join(cnn_model_dir, 'scaler_nn.pkl')
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            self.logger.info(f"Scaler loaded for CNN models: {scaler_file}")
        else:
            self.logger.error(f"Scaler file not found for CNN models: {scaler_file}")
            scaler = None
        # Load CNN models and associate the scaler
        models['cnn']['models'] = [load_model(file) for file in cnn_model_files]
        models['cnn']['scaler'] = scaler  # Single scaler for all CNN models
        self.logger.info(f"{len(models['cnn']['models'])} CNN models loaded.")

        # CNN-LSTM models
        cnn_lstm_model_dir = os.path.join(models_dir, 'cnn_lstm_model_combined')
        cnn_lstm_model_files = [
            os.path.join(cnn_lstm_model_dir, file)
            for file in os.listdir(cnn_lstm_model_dir) if file.endswith('.keras')
        ]
        # Load scaler for CNN-LSTM models
        scaler_file = os.path.join(cnn_lstm_model_dir, 'scaler_nn.pkl')
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            self.logger.info(f"Scaler loaded for CNN-LSTM models: {scaler_file}")
        else:
            self.logger.error(f"Scaler file not found for CNN-LSTM models: {scaler_file}")
            scaler = None
        # Load CNN-LSTM models and associate the scaler
        models['cnn_lstm']['models'] = [load_model(file) for file in cnn_lstm_model_files]
        models['cnn_lstm']['scaler'] = scaler
        self.logger.info(f"{len(models['cnn_lstm']['models'])} CNN-LSTM models loaded.")

        # LSTM models
        lstm_model_dir = os.path.join(models_dir, 'lstm_model_combined')
        lstm_model_files = [
            os.path.join(lstm_model_dir, file)
            for file in os.listdir(lstm_model_dir) if file.endswith('.keras')
        ]
        # Load scaler for LSTM models
        scaler_file = os.path.join(lstm_model_dir, 'scaler_nn.pkl')
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            self.logger.info(f"Scaler loaded for LSTM models: {scaler_file}")
        else:
            self.logger.error(f"Scaler file not found for LSTM models: {scaler_file}")
            scaler = None
        # Load LSTM models and associate the scaler
        models['lstm']['models'] = [load_model(file) for file in lstm_model_files]
        models['lstm']['scaler'] = scaler
        self.logger.info(f"{len(models['lstm']['models'])} LSTM models loaded.")

        # Supervised models
        supervised_model_dir = os.path.join(models_dir, 'anomaly_detection_model_combined')
        supervised_model_files = [
            os.path.join(supervised_model_dir, file)
            for file in os.listdir(supervised_model_dir) if file.endswith('best_model.pkl') and 'model' in file
        ]
        # Load scaler for supervised models
        scaler_file = os.path.join(supervised_model_dir, 'scaler.pkl')
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            self.logger.info(f"Scaler loaded for supervised models: {scaler_file}")
        else:
            self.logger.error(f"Scaler file not found for supervised models: {scaler_file}")
            scaler = None
        # Load supervised models and associate the scaler
        models['supervised']['models'] = [joblib.load(file) for file in supervised_model_files]
        models['supervised']['scaler'] = scaler
        self.logger.info(f"{len(models['supervised']['models'])} supervised models loaded.")

        self.logger.info("All models and scalers loaded successfully.")

    def load_config(self):
        config_file = self.config_file
        if not os.path.exists(config_file):
            self.logger.error(f"Configuration file does not exist: {config_file}")
            self.config = None
            return
        with open(config_file, 'r') as file:
            try:
                self.config = yaml.safe_load(file)
                self.logger.info(f"Configuration loaded successfully from {config_file}")
            except yaml.YAMLError as e:
                self.logger.error(f"Error loading configuration file: {e}")
                self.config = None

    def create_database_engine(self):
        config = self.config
        if config is None or 'database' not in config or 'url' not in config['database']:
            self.logger.error("Database configuration is missing or invalid")
            self.engine = None
            return
        try:
            self.engine = create_engine(config['database']['url'])
            self.logger.info("Database engine created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database engine: {e}")
            self.engine = None

    def save_to_database(self, data, table_name):
        engine = self.engine
        if engine is None:
            self.logger.error("Engine is not initialized")
            return
        try:
            data.to_sql(table_name, engine, if_exists='append', index=False)
            self.logger.info(f"Data saved to database table {table_name} successfully")
        except Exception as e:
            self.logger.error(f"Error saving data to database: {e}")

    def feature_engineering(self, df):
        # Log the received data
        self.logger.debug(f"Data before feature engineering:\n{df.to_dict()}")

        # Check if 'Type' column is present
        if 'Type' not in df.columns:
            self.logger.error("Type column is missing from the input data.")
            return None

        # Validate data types for all required fields
        expected_types = {
            'Air temperature [K]': (int, float),
            'Process temperature [K]': (int, float),
            'Rotational speed [rpm]': (int, float),
            'Torque [Nm]': (int, float),
            'Tool wear [min]': (int, float),
            'Type': str
        }

        for column, expected in expected_types.items():
            if column not in df.columns:
                self.logger.error(f"Missing required column: {column}")
                return None
            if not df[column].map(lambda x: isinstance(x, expected)).all():
                self.logger.error(f"Invalid data type in column '{column}'. Expected types: {expected}")
                return None

        # Encode 'Type' using the reconstructed mapping
        df['Type'] = df['Type'].map(self.type_mapping)
        if df['Type'].isnull().any():
            self.logger.error(f"Unknown Type encountered: {df['Type'].unique()}")
            return None  # Skip processing this record

        # Map Type to thresholds
        thresholds = {0: 13000, 1: 11000, 2: 12000}
        df['OSF_threshold'] = df['Type'].map(thresholds)

        # Calculate additional features
        df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
        df['Rotational speed [rad/s]'] = df['Rotational speed [rpm]'] * (2 * np.pi / 60)
        df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rad/s]']
        df['Tool_Torque_Product'] = df['Tool wear [min]'] * df['Torque [Nm]']

        # Failure condition features
        df['TWF_condition'] = (
                (df['Tool wear [min]'] >= 200) & (df['Tool wear [min]'] <= 240)
        ).astype(int)
        df['HDF_condition'] = (
                (df['Temp_diff'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)
        ).astype(int)
        df['PWF_condition'] = (
                (df['Power'] < 3500) | (df['Power'] > 9000)
        ).astype(int)
        df['OSF_condition'] = (
                df['Tool_Torque_Product'] > df['OSF_threshold']
        ).astype(int)

        # Aggregate failure risk
        df['Failure_Risk'] = (
                df['TWF_condition'] |
                df['HDF_condition'] |
                df['PWF_condition'] |
                df['OSF_condition']
        ).astype(int)

        # Log the features after engineering
        self.logger.debug(f"Features after feature engineering:\n{df.to_dict()}")
        return df

    def prepare_data(self, df):
        # Ensure all necessary features are present
        required_features = self.selected_features.copy()
        for feature in self.features_nn:
            if feature not in required_features:
                required_features.append(feature)
        if not set(required_features).issubset(df.columns):
            missing_features = set(required_features) - set(df.columns)
            self.logger.error(f"Missing required features: {missing_features}")
            return None
        self.logger.debug(f"Data after preparation:\n{df[required_features].to_dict()}")
        return df

    def aggregate_predictions(self, predictions):
        # Simple averaging (weights can be adjusted)
        final_score = np.mean(predictions)
        final_prediction = 1 if final_score >= 0.5 else 0
        return final_prediction, final_score

    def setup_kafka_consumer(self, retries=5, delay=2):
        attempt = 0
        while attempt < retries:
            try:
                self.consumer = KafkaConsumer(
                    'sensor-data',
                    bootstrap_servers=self.bootstrap_servers,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    group_id='sensor-data-group',
                    consumer_timeout_ms=1000 * 20  # Timeout after 20 seconds of inactivity
                )
                self.logger.info("Kafka consumer connected.")
                return
            except Exception as e:
                self.logger.error(f"Kafka consumer setup error: {e}")
                attempt += 1
                self.logger.info(f"Retrying Kafka consumer setup in {delay} seconds... (Attempt {attempt}/{retries})")
                time.sleep(delay)
        self.logger.error("Failed to connect Kafka consumer after multiple attempts.")
        self.consumer = None

    def setup_kafka_producer(self, retries=5, delay=2):
        attempt = 0
        while attempt < retries:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda x: json.dumps(x).encode('utf-8')
                )
                self.logger.info("Kafka producer connected.")
                return
            except Exception as e:
                self.logger.error(f"Kafka producer setup error: {e}")
                attempt += 1
                self.logger.info(f"Retrying Kafka producer setup in {delay} seconds... (Attempt {attempt}/{retries})")
                time.sleep(delay)
        self.logger.error("Failed to connect Kafka producer after multiple attempts.")
        self.producer = None


    def process_messages(self):
        # Feature mapping for supervised models (temporary fix)
        feature_mapping = {
            'Column_0': 'Air temperature [K]',
            'Column_1': 'Process temperature [K]',
            'Column_2': 'Rotational speed [rpm]',
            'Column_3': 'Torque [Nm]',
            'Column_4': 'Tool wear [min]',
            'Column_5': 'Temp_diff',
            'Column_6': 'Rotational speed [rad/s]',
            'Column_7': 'Power',
            'Column_8': 'Tool_Torque_Product',
            'Column_9': 'TWF_condition',
            'Column_10': 'HDF_condition',
            'Column_11': 'PWF_condition',
            'Column_12': 'Failure_Risk',
            'Column_13': 'Type',
            'Column_14': 'OSF_condition'
        }

        try:
            # Process incoming data
            for message in self.consumer:
                try:
                    data = message.value
                    df = pd.DataFrame([data])

                    # Log the received data
                    self.logger.debug(f"Received data:\n{df.to_dict()}")

                    # Feature engineering
                    df = self.feature_engineering(df)
                    if df is None:
                        self.logger.error("Feature engineering failed. Skipping this record.")
                        continue  # Skip this message due to error

                    # Prepare data
                    df_processed = self.prepare_data(df)
                    if df_processed is None:
                        self.logger.error("Data preparation failed. Skipping this record.")
                        continue  # Skip this message due to error

                    predictions = []

                    # Supervised models
                    scaler_supervised = self.models['supervised']['scaler']
                    for model in self.models['supervised']['models']:
                        # Retrieve model's expected feature names
                        if hasattr(model, 'feature_name_'):
                            model_feature_names = model.feature_name_
                            self.logger.debug(f"Supervised model feature names: {model_feature_names}")
                        elif hasattr(model, 'booster_'):
                            model_feature_names = model.booster_.feature_name()
                            self.logger.debug(f"Supervised model booster feature names: {model_feature_names}")
                        else:
                            self.logger.error("Supervised model does not have feature names attribute.")
                            continue

                        # Ensure all expected features are in the input data
                        if hasattr(scaler_supervised, 'feature_names_in_'):
                            scaler_feature_names = scaler_supervised.feature_names_in_
                            self.logger.debug(f"Supervised scaler feature names: {scaler_feature_names}")
                        else:
                            self.logger.error("Supervised scaler does not have feature names attribute.")
                            continue

                        missing_features = set(scaler_feature_names) - set(df_processed.columns)
                        if missing_features:
                            self.logger.error(f"Missing required features for scaler: {missing_features}")
                            continue

                        # Subset df_processed to scaler's expected features
                        df_supervised = df_processed[scaler_feature_names]
                        self.logger.debug(f"DataFrame columns before scaling: {df_supervised.columns.tolist()}")

                        # Transform the data
                        if scaler_supervised is not None:
                            X_scaled = scaler_supervised.transform(df_supervised)
                            self.logger.debug(f"Data scaled using scaler.")
                        else:
                            X_scaled = df_supervised.values
                            self.logger.debug(f"Data not scaled, using raw values.")

                        # Create DataFrame from scaled data
                        df_scaled = pd.DataFrame(X_scaled, columns=scaler_feature_names)
                        self.logger.debug(f"DataFrame columns after scaling: {df_scaled.to_dict()}")

                        # Rename columns to model's expected feature names
                        reverse_feature_mapping = {v: k for k, v in feature_mapping.items()}
                        df_scaled = df_scaled.rename(columns=reverse_feature_mapping)
                        self.logger.debug(f"DataFrame columns after renaming: {df_scaled.to_dict()}")

                        # Ensure all model features are in df_scaled
                        missing_features = set(model_feature_names) - set(df_scaled.columns)
                        if missing_features:
                            self.logger.error(f"Missing required features for model: {missing_features}")
                            continue

                        # Subset df_scaled to model_feature_names
                        df_model_input = df_scaled[model_feature_names]
                        self.logger.debug(f"DataFrame columns for model input: {df_model_input.to_dict()}")

                        # Predict
                        pred = model.predict(df_model_input)[0]
                        predictions.append(pred)
                        self.logger.info(f"Supervised model predicted: {pred}")

                    # Neural network models
                    for model_type in ['cnn', 'lstm', 'cnn_lstm']:
                        scaler_nn = self.models[model_type]['scaler']

                        if scaler_nn is not None and hasattr(scaler_nn, 'feature_names_in_'):
                            scaler_feature_names = scaler_nn.feature_names_in_
                            self.logger.debug(f"Scaler feature names for {model_type}: {scaler_feature_names}")

                            missing_features = set(scaler_feature_names) - set(df_processed.columns)
                            if missing_features:
                                self.logger.error(f"Missing required features for scaler {model_type}: {missing_features}")
                                continue  # Skip this iteration if required features are missing

                            # Reindex df_processed to match the scaler's expected feature names and order
                            df_nn = df_processed[scaler_feature_names]
                        else:
                            self.logger.error(f"Scaler for {model_type} does not have feature_names_in_.")
                            continue  # Skip this iteration if scaler doesn't have feature names

                        self.logger.debug(f"Data for {model_type} models before scaling:\n{df_nn.to_dict()}")

                        # Proceed with scaling
                        try:
                            X_nn_scaled = scaler_nn.transform(df_nn)
                            self.logger.debug(f"Data scaled using scaler for {model_type} models.")
                        except Exception as e:
                            self.logger.error(f"Error during scaling for {model_type} models: {e}")
                            continue  # Skip this iteration if scaling fails

                        # Update sequence buffer for this model type
                        sequence_buffer = self.nn_sequence_buffers[model_type]
                        sequence_buffer.append(X_nn_scaled[0])
                        if len(sequence_buffer) > self.sequence_length:
                            sequence_buffer.pop(0)
                        self.logger.debug(f"Sequence buffer for {model_type} models: {sequence_buffer}")

                        if len(sequence_buffer) >= self.sequence_length:
                            X_sequence = np.array(sequence_buffer).reshape(1, self.sequence_length, X_nn_scaled.shape[1])
                            if model_type == 'cnn_lstm':
                                X_sequence = X_sequence.reshape(1, self.sequence_length, X_nn_scaled.shape[1], 1)
                            # Predict with all models of this type
                            for model in self.models[model_type]['models']:
                                pred = model.predict(X_sequence, verbose=0)[0][0]
                                predictions.append(pred)
                                self.logger.info(f"{model_type.upper()} model predicted: {pred}")

                    # Aggregate predictions
                    final_prediction, final_score = self.aggregate_predictions(predictions)
                    self.logger.info(f"Aggregated prediction: {final_prediction}, Score: {final_score:.4f}")

                    # Add predictions to the processed dataframe
                    df_processed['final_prediction'] = final_prediction
                    df_processed['final_score'] = final_score
                    df_processed['timestamp'] = datetime.now(timezone.utc)

                    if final_score >= 0.6:
                        # Send prediction to Kafka
                        prediction_message = {
                            'PredictedFailureDate': df_processed['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
                            'RiskScore': float(final_score)
                        }

                        try:
                            self.producer.send('failure_predictions', value=prediction_message)
                            self.producer.flush()
                            self.logger.info(f"Prediction sent to Kafka: {prediction_message}")
                        except Exception as e:
                            self.logger.error(f"Failed to send prediction to Kafka: {e}")

                    # Save results to database
                    self.save_to_database(df_processed, 'real_time_predictions')

                    self.logger.info(f"Data processed and saved. Prediction: {final_prediction}, Score: {final_score:.4f}")

                except Exception as e:
                    self.logger.error(f"Error processing message: {e}", exc_info=True)
                    continue

        except StopIteration:
            self.logger.info("No more messages. Exiting consumer.")
        except KeyboardInterrupt:
            self.logger.info("Consumer interrupted by user.")
        except Exception as e:
            self.logger.error(f"Error in get message: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            if self.consumer is not None:
                self.consumer.close()
                self.logger.info("Kafka consumer closed.")
            if self.producer is not None:
                self.producer.close()
                self.logger.info("Kafka producer closed.")
            if self.engine is not None:
                self.engine.dispose()
                self.logger.info("Database engine disposed.")
        except Exception as e:
            self.logger.error(f"Error while cleanup RealTimeProcessorClient : {e}")

        try:
            # Log before closing handlers
            self.logger.info("RealTimeProcessorClient is shutting down.")

            # Close and remove all handlers associated with the logger
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)
                self.logger.debug(f"Closed and removed handler: {handler}")
        except Exception as e:
            print(f"Error while closing logger: {e}")

