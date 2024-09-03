import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils import resample

# Setup logging
log_file_path = '../../logs/model_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])

def create_lstm_model(input_shape, learning_rate=0.001, dropout_rate=0.2, lstm_units=50):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    logging.info(
        f"LSTM model created successfully with {lstm_units} units, dropout rate {dropout_rate}, and learning rate {learning_rate}")
    return model

def train_lstm_on_fold(X_train, y_train, X_val, y_val, config, fold, model_dir, db_name):
    lstm_model = create_lstm_model((X_train.shape[1], 1), learning_rate=config['learning_rate'],
                                   dropout_rate=config['dropout_rate'], lstm_units=config['lstm_units'])
    history_lstm = lstm_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=config['epochs'],
                                  batch_size=config['batch_size'],
                                  validation_data=(np.expand_dims(X_val, axis=-1), y_val))

    lstm_model_path = os.path.join(model_dir, f"{db_name}_fold_{fold}.keras")
    lstm_history_path = os.path.join(model_dir, f"{db_name}_fold_{fold}_history.csv")
    lstm_model.save(lstm_model_path)
    pd.DataFrame(history_lstm.history).to_csv(lstm_history_path)
    logging.info(f"LSTM model and history saved for fold {fold} of {db_name}")

    return history_lstm

def cross_validate_lstm(X, y, config, timestamp, db_name):
    skf = StratifiedKFold(n_splits=5)
    model_dir = os.path.join(config['model_save_path'], f'lstm_model_{db_name}')
    os.makedirs(model_dir, exist_ok=True)

    aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"Training on fold {fold}/5 for {db_name}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        history_lstm = train_lstm_on_fold(X_train, y_train, X_val, y_val, config, fold, model_dir, db_name)
        auc = max(history_lstm.history['val_AUC'])
        aucs.append(auc)

    avg_auc = np.mean(aucs)
    logging.info(f"Average AUC across folds for {db_name}: {avg_auc}")

    if avg_auc < 0.7 or max(aucs) - min(aucs) > 0.1:  # Thresholds to detect poor performance or overfitting
        logging.info(f"Low AUC detected or high variance between folds for {db_name}, adjusting hyperparameters...")
        config['learning_rate'] *= 0.5  # Reduce learning rate
        config['lstm_units'] += 32  # Increase LSTM units
        config['dropout_rate'] += 0.1  # Increase dropout rate
        logging.info(
            f"New parameters for {db_name} - Learning Rate: {config['learning_rate']}, LSTM Units: {config['lstm_units']}, Dropout Rate: {config['dropout_rate']}")
        cross_validate_lstm(X, y, config, timestamp, db_name)
    else:
        logging.info(f"Model performance for {db_name} is satisfactory.")

# Example usage for all three databases:
if __name__ == "__main__":
    config = {
        'model_save_path': '../../models',
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'lstm_units': 64,
        'feature_columns': ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                            'Tool wear [min]', 'Air temperature [K]_lag_1', 'Air temperature [K]_lag_2',
                            'Air temperature [K]_lag_3', 'Air temperature [K]_lag_4', 'Air temperature [K]_lag_5',
                            'Process temperature [K]_lag_1', 'Process temperature [K]_lag_2', 'Process temperature [K]_lag_3',
                            'Process temperature [K]_lag_4', 'Process temperature [K]_lag_5', 'Rotational speed [rpm]_lag_1',
                            'Rotational speed [rpm]_lag_2', 'Rotational speed [rpm]_lag_3', 'Rotational speed [rpm]_lag_4',
                            'Rotational speed [rpm]_lag_5', 'Torque [Nm]_lag_1', 'Torque [Nm]_lag_2',
                            'Torque [Nm]_lag_3', 'Torque [Nm]_lag_4', 'Torque [Nm]_lag_5', 'Tool wear [min]_lag_1',
                            'Tool wear [min]_lag_2', 'Tool wear [min]_lag_3', 'Tool wear [min]_lag_4', 'Tool wear [min]_lag_5'],
        'target_column': 'Machine failure'
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    databases = {
        'simulation': '../../data/prepare_lstm_training_data/simulation_X_train.csv',
        'combined': '../../data/prepare_lstm_training_data/combined_X_train.csv',
        'real': '../../data/prepare_lstm_training_data/processed_X_train.csv'
    }

    for db_name, db_path in databases.items():
        logging.info(f"Processing database: {db_name}")
        X = pd.read_csv(db_path)
        y_path = db_path.replace('X_train', 'y_train')
        y = pd.read_csv(y_path).values.flatten()

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=config['feature_columns'])

        # Balance the dataset
        data_balanced = pd.concat([X_scaled, pd.Series(y, name=config['target_column'])], axis=1)
        majority = data_balanced[data_balanced[config['target_column']] == 0]
        minority = data_balanced[data_balanced[config['target_column']] == 1]
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        balanced_data = pd.concat([majority, minority_upsampled])
        X_balanced = balanced_data[config['feature_columns']]
        y_balanced = balanced_data[config['target_column']]

        # Cross-validate and adjust hyperparameters automatically
        cross_validate_lstm(X_balanced.values, y_balanced.values, config, timestamp, db_name)
