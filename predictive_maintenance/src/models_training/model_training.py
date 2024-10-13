import os
import pandas as pd
import numpy as np
import joblib
import logging
import yaml
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Input
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info("Configuration loaded successfully from %s", config_path)
    return config

# Load data
def load_data(train_path, val_path):
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    logging.info("Training and validation data loaded successfully")
    return train_data, val_data

# Preprocess data
def preprocess_data(train_data, val_data, feature_columns, target_column):
    scaler = StandardScaler()
    numeric_columns = train_data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()

    X_train = train_data[numeric_columns]
    y_train = train_data[target_column]
    X_val = val_data[numeric_columns]
    y_val = val_data[target_column]

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logging.info("Data preprocessing completed successfully")
    return X_train_scaled, y_train, X_val_scaled, y_val, scaler

# Check class distribution
def check_class_distribution(y_train, y_val):
    from collections import Counter
    counter_train = Counter(y_train)
    counter_val = Counter(y_val)
    logging.info(f"Training class distribution: {counter_train}")
    logging.info(f"Validation class distribution: {counter_val}")
    return counter_train, counter_val

# Oversample the minority class in the training data
def balance_dataset(X_train, y_train):
    from sklearn.utils import resample
    X = pd.concat([pd.DataFrame(X_train), y_train.reset_index(drop=True)], axis=1)
    minority = X[X[y_train.name] == 1]
    majority = X[X[y_train.name] == 0]

    minority_upsampled = resample(minority,
                                  replace=True,  # sample with replacement
                                  n_samples=len(majority),  # match number in majority class
                                  random_state=42)  # reproducible results

    upsampled = pd.concat([majority, minority_upsampled])
    X_train_balanced = upsampled.drop(y_train.name, axis=1)
    y_train_balanced = upsampled[y_train.name]

    return X_train_balanced, y_train_balanced

# Define LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    logging.info("LSTM model created successfully")
    return model

# Define CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    logging.info("CNN model created successfully")
    return model

# Define Autoencoder model
def create_autoencoder_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(input_shape[0], activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("Autoencoder model created successfully")
    return model

# Train and save LSTM model
def train_and_save_lstm_model(X_train, y_train, X_val, y_val, config, timestamp):
    lstm_model_path = f"{config['lstm_model_path']}_{timestamp}.keras"
    lstm_history_path = f"{config['lstm_history_path']}_{timestamp}.csv"
    if not os.path.exists(lstm_model_path):
        lstm_model = create_lstm_model((X_train.shape[1], 1))
        history_lstm = lstm_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=config['epochs'],
                                      batch_size=config['batch_size'],
                                      validation_data=(np.expand_dims(X_val, axis=-1), y_val))
        lstm_model.save(lstm_model_path)
        pd.DataFrame(history_lstm.history).to_csv(lstm_history_path)
        logging.info("LSTM_model trained and saved successfully")
    else:
        logging.info("LSTM_model already exists. Skipping training.")

# Train and save CNN model
def train_and_save_cnn_model(X_train, y_train, X_val, y_val, config, timestamp):
    cnn_model_path = f"{config['cnn_model_path']}_{timestamp}.keras"
    cnn_history_path = f"{config['cnn_history_path']}_{timestamp}.csv"
    if not os.path.exists(cnn_model_path):
        cnn_model = create_cnn_model((X_train.shape[1], 1))
        history_cnn = cnn_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=config['epochs'],
                                    batch_size=config['batch_size'],
                                    validation_data=(np.expand_dims(X_val, axis=-1), y_val))
        cnn_model.save(cnn_model_path)
        pd.DataFrame(history_cnn.history).to_csv(cnn_history_path)
        logging.info("CNN_model trained and saved successfully")
    else:
        logging.info("CNN_model already exists. Skipping training.")

# Train and save Autoencoder model
def train_and_save_autoencoder_model(X_train, y_train, X_val, y_val, config, timestamp):
    autoencoder_model_path = f"{config['autoencoder_model_path']}_{timestamp}.keras"
    autoencoder_history_path = f"{config['autoencoder_history_path']}_{timestamp}.csv"
    if not os.path.exists(autoencoder_model_path):
        autoencoder_model = create_autoencoder_model((X_train.shape[1],))
        history_autoencoder = autoencoder_model.fit(X_train, X_train, epochs=config['epochs'],
                                                    batch_size=config['batch_size'], validation_data=(X_val, X_val))
        autoencoder_model.save(autoencoder_model_path)
        pd.DataFrame(history_autoencoder.history).to_csv(autoencoder_history_path)
        logging.info("Autoencoder_model trained and saved successfully")
    else:
        logging.info("Autoencoder_model already exists. Skipping training.")



# Function to train and save LSTM model
def train_and_save_lstm_model(X_train, y_train, X_val, y_val, config, timestamp):
    lstm_model_path = f"{config['lstm_model_path']}_{timestamp}.keras"
    lstm_history_path = f"{config['lstm_history_path']}_{timestamp}.csv"

    if not os.path.exists(lstm_model_path):
        lstm_model = create_lstm_model((X_train.shape[1], 1))
        history_lstm = lstm_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=config['epochs'],
                                      batch_size=config['batch_size'],
                                      validation_data=(np.expand_dims(X_val, axis=-1), y_val))
        lstm_model.save(lstm_model_path)
        pd.DataFrame(history_lstm.history).to_csv(lstm_history_path)
        logging.info("LSTM_model trained and saved successfully")
    else:
        logging.info("LSTM_model already exists. Skipping training.")

# Function to train and save CNN model
def train_and_save_cnn_model(X_train, y_train, X_val, y_val, config, timestamp):
    cnn_model_path = f"{config['cnn_model_path']}_{timestamp}.keras"
    cnn_history_path = f"{config['cnn_history_path']}_{timestamp}.csv"

    if not os.path.exists(cnn_model_path):
        cnn_model = create_cnn_model((X_train.shape[1], 1))
        history_cnn = cnn_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=config['epochs'],
                                    batch_size=config['batch_size'],
                                    validation_data=(np.expand_dims(X_val, axis=-1), y_val))
        cnn_model.save(cnn_model_path)
        pd.DataFrame(history_cnn.history).to_csv(cnn_history_path)
        logging.info("CNN_model trained and saved successfully")
    else:
        logging.info("CNN_model already exists. Skipping training.")

# Function to train and save Autoencoder model
def train_and_save_autoencoder_model(X_train, y_train, X_val, y_val, config, timestamp):
    autoencoder_model_path = f"{config['autoencoder_model_path']}_{timestamp}.keras"
    autoencoder_history_path = f"{config['autoencoder_history_path']}_{timestamp}.csv"

    if not os.path.exists(autoencoder_model_path):
        autoencoder_model = create_autoencoder_model((X_train.shape[1],))
        history_autoencoder = autoencoder_model.fit(X_train, X_train, epochs=config['epochs'],
                                                    batch_size=config['batch_size'], validation_data=(X_val, X_val))
        autoencoder_model.save(autoencoder_model_path)
        pd.DataFrame(history_autoencoder.history).to_csv(autoencoder_history_path)
        logging.info("Autoencoder_model trained and saved successfully")
    else:
        logging.info("Autoencoder_model already exists. Skipping training.")

# Function to train and save Isolation Forest model
def train_and_save_isolation_forest_model(X_train, y_train, X_val, y_val, config, timestamp):
    isolation_forest_model_path = f"{config['isolation_forest_model_path']}_{timestamp}.joblib"

    if not os.path.exists(isolation_forest_model_path):
        isolation_forest_model = IsolationForest(contamination=0.1, random_state=42)
        isolation_forest_model.fit(X_train)
        joblib.dump(isolation_forest_model, isolation_forest_model_path)
        logging.info("Isolation Forest model trained and saved successfully")
    else:
        logging.info("Isolation Forest model already exists. Skipping training.")

# Evaluate model and add more metrics
def evaluate_model(model, X_val, y_val, threshold=0.5):
    logging.info(f"Shape of X_val before prediction: {X_val.shape}")
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc,
        'auc': auc
    }

    for metric, value in results.items():
        logging.info(f"{metric.capitalize()}: {value}")

    return results

# Main script execution
if __name__ == "__main__":
    config_path = "../config/model_config.yaml"
    config = load_config(config_path)

    train_path = config['train_data_path']
    val_path = config['val_data_path']
    train_data, val_data = load_data(train_path, val_path)

    X_train, y_train, X_val, y_val, scaler = preprocess_data(train_data, val_data, config['feature_columns'], config['target_column'])

    # Balance the training dataset
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # Check class distribution
    check_class_distribution(y_train_balanced, y_val)

    # Train and save models in parallel
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(train_and_save_lstm_model, X_train_balanced, y_train_balanced, X_val, y_val, config, timestamp))
        futures.append(executor.submit(train_and_save_cnn_model, X_train_balanced, y_train_balanced, X_val, y_val, config, timestamp))
        futures.append(executor.submit(train_and_save_autoencoder_model, X_train_balanced, y_train_balanced, X_val, y_val, config, timestamp))
        futures.append(executor.submit(train_and_save_isolation_forest_model, X_train_balanced, y_train_balanced, X_val, y_val, config, timestamp))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Exception occurred: {e}")

    # Evaluate models
    lstm_model = load_model(f"{config['lstm_model_path']}_{timestamp}.keras")
    lstm_results = evaluate_model(lstm_model, np.expand_dims(X_val, axis=-1), y_val)

    cnn_model = load_model(f"{config['cnn_model_path']}_{timestamp}.keras")
    cnn_results = evaluate_model(cnn_model, np.expand_dims(X_val, axis=-1), y_val)

    autoencoder_model = load_model(f"{config['autoencoder_model_path']}_{timestamp}.keras")
    autoencoder_results = evaluate_model(autoencoder_model, X_val, X_val)  # Autoencoder's target is the input itself

    # Log the evaluation results
    logging.info("Model evaluation results:")
    logging.info(f"LSTM Model Results: {lstm_results}")
    logging.info(f"CNN Model Results: {cnn_results}")
    logging.info(f"Autoencoder Model Results: {autoencoder_results}")

    # Save evaluation results
    results_path = config['results_path']
    os.makedirs(results_path, exist_ok=True)
    pd.DataFrame([lstm_results, cnn_results, autoencoder_results],
                 index=['LSTM', 'CNN', 'Autoencoder']).to_csv(os.path.join(results_path, 'evaluation_results.csv'))

    logging.info("Model evaluation results saved successfully")
    logging.info("Model training process completed")
