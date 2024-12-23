import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE  # Using SMOTE for imbalanced dataset handling

# Setup logging
log_file_path = '../../../logs/model_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])

# CNN-LSTM model creation function with L2 regularization, Batch Normalization, and adaptive dropout
def create_cnn_lstm_model(input_shape, learning_rate=0.001, dropout_rate=0.3, filters=64, lstm_units=50, l2_reg=0.001):
    model = Sequential()

    # Dynamically set kernel size based on the number of time steps (input_shape[0])
    time_steps = input_shape[0]
    kernel_size = min(3, time_steps)  # Ensure the kernel size doesn't exceed the number of time steps

    # CNN layers with dynamic kernel size
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape,
                     kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())  # Batch Normalization after Conv1D
    if time_steps > 1:
        model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())  # Batch Normalization after Conv1D
    if time_steps > 1:
        model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    # LSTM layers with L2 regularization
    model.add(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())  # Batch Normalization after LSTM
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())  # Batch Normalization after LSTM
    model.add(Dropout(dropout_rate))

    # Dense output layer
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    logging.info(
        f"CNN-LSTM model created with {filters} filters, {lstm_units} LSTM units, dropout rate {dropout_rate}, learning rate {learning_rate}, and kernel size {kernel_size}")
    return model

# Function to evaluate model performance
def evaluate_model_performance(y_true, y_pred, db_name, fold):
    cm = confusion_matrix(y_true, np.round(y_pred))
    f1 = f1_score(y_true, np.round(y_pred))
    logging.info(f"Confusion Matrix for {db_name}, Fold {fold}:\n{cm}")
    logging.info(f"F1-Score for {db_name}, Fold {fold}: {f1}")
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    logging.info(f"Precision-Recall AUC for {db_name}, Fold {fold}: {pr_auc}")
    return f1, pr_auc

# Function to train CNN-LSTM on a single fold
def train_cnn_lstm_on_fold(X_train, y_train, X_val, y_val, config, fold, model_dir, db_name):
    cnn_lstm_model = create_cnn_lstm_model((X_train.shape[1], X_train.shape[2]), learning_rate=config['learning_rate'],
                                           dropout_rate=config['dropout_rate'], filters=config['filters'],
                                           lstm_units=config['lstm_units'], l2_reg=config['l2_reg'])

    # Early stopping on validation loss to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min')
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

    history_cnn_lstm = cnn_lstm_model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                                          validation_data=(X_val, y_val), callbacks=[early_stopping, lr_reduction])

    model_path = os.path.join(model_dir, f"{db_name}_fold_{fold}.keras")
    history_path = os.path.join(model_dir, f"{db_name}_fold_{fold}_history.csv")
    cnn_lstm_model.save(model_path)
    pd.DataFrame(history_cnn_lstm.history).to_csv(history_path)
    logging.info(f"CNN-LSTM model and history saved for fold {fold} of {db_name}")

    y_pred_val = cnn_lstm_model.predict(X_val)
    f1, pr_auc = evaluate_model_performance(y_val, y_pred_val, db_name, fold)
    return history_cnn_lstm, f1, pr_auc

# Cross-validation function for CNN-LSTM with stratified cross-validation
def cross_validate_cnn_lstm(X, y, config, timestamp, db_name, quick_evaluation=False):
    skf = StratifiedKFold(n_splits=5)
    model_dir = os.path.join(config['model_save_path'], f'cnn_lstm_model_{db_name}')
    os.makedirs(model_dir, exist_ok=True)

    aucs = []
    f1_scores = []
    pr_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"Training on fold {fold}/5 for {db_name}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        history_cnn_lstm, f1, pr_auc = train_cnn_lstm_on_fold(X_train, y_train, X_val, y_val, config, fold, model_dir, db_name)
        auc = max(history_cnn_lstm.history['val_AUC'])
        aucs.append(auc)
        f1_scores.append(f1)
        pr_aucs.append(pr_auc)

    avg_auc = np.mean(aucs)
    avg_f1 = np.mean(f1_scores)
    avg_pr_auc = np.mean(pr_aucs)

    logging.info(f"Average AUC across folds for {db_name}: {avg_auc}")
    logging.info(f"Average F1-Score across folds for {db_name}: {avg_f1}")
    logging.info(f"Average Precision-Recall AUC across folds for {db_name}: {avg_pr_auc}")

    return avg_f1

# Function to select optimal time steps and features
def select_optimal_time_steps_and_features(X, y, config, db_name, timestamp):
    best_time_steps = None
    best_f1_score = 0
    best_X_reshaped = None

    # Ensure X is 2D before reshaping for different time steps
    if X.ndim == 3:
        logging.info(f"Input data is already 3D: {X.shape}")
        return X
    elif X.ndim == 2:
        time_steps_range = [2, 3, 5, 10]

        for time_steps in time_steps_range:
            num_samples, num_features = X.shape
            if num_features % time_steps != 0:
                continue  # Skip if the number of features can't be evenly divided by time steps

            num_features_per_step = num_features // time_steps
            X_reshaped = X.reshape(num_samples, time_steps, num_features_per_step)

            # Perform a quick evaluation using cross-validation to find the best configuration
            avg_f1 = cross_validate_cnn_lstm(X_reshaped, y, config, timestamp, db_name, quick_evaluation=True)

            if avg_f1 > best_f1_score:
                best_f1_score = avg_f1
                best_time_steps = time_steps
                best_X_reshaped = X_reshaped

        logging.info(f"Best time steps: {best_time_steps}, with F1 score: {best_f1_score}")
        return best_X_reshaped

    else:
        raise ValueError(f"Unexpected input shape: {X.shape}. Expected 2D or 3D data.")

# Main script to train CNN-LSTM model on all datasets
if __name__ == "__main__":
    config = {
        'model_save_path': '../../../models',
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'filters': 64,
        'lstm_units': 50,
        'l2_reg': 0.01,  # L2 regularization parameter
        'feature_columns': ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                            'Tool wear [min]', 'Air temperature [K]_lag_1', 'Air temperature [K]_lag_2',
                            'Air temperature [K]_lag_3', 'Air temperature [K]_lag_4', 'Air temperature [K]_lag_5',
                            'Process temperature [K]_lag_1', 'Process temperature [K]_lag_2', 'Process temperature [K]_lag_3',
                            'Process temperature [K]_lag_4', 'Process temperature [K]_lag_5', 'Rotational speed [rpm]_lag_1',
                            'Rotational speed [rpm]_lag_2', 'Rotational speed [rpm]_lag_3', 'Rotational speed [rpm]_lag_4',
                            'Rotational speed [rpm]_lag_5', 'Torque [Nm]_lag_1', 'Torque [Nm]_lag_2', 'Torque [Nm]_lag_3',
                            'Torque [Nm]_lag_4', 'Torque [Nm]_lag_5', 'Tool wear [min]_lag_1', 'Tool wear [min]_lag_2',
                            'Tool wear [min]_lag_3', 'Tool wear [min]_lag_4', 'Tool wear [min]_lag_5'],
        'target_column': 'Machine failure'
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    databases = {
        'simulation': '../../../data/prepare_cnn_lstm_training_data/simulation_X_train.npy',
        'combined': '../../../data/prepare_cnn_lstm_training_data/combined_X_train.npy',
        'real': '../../../data/prepare_cnn_lstm_training_data/processed_X_train.npy'
    }

    for db_name, db_path in databases.items():
        logging.info(f"Processing database: {db_name}")

        # Load the feature and target data
        X = np.load(db_path)
        y_path = db_path.replace('X_train', 'y_train')
        y = np.load(y_path)

        logging.info(f"Loaded X shape: {X.shape}")

        # Automatically select optimal time steps and features
        X_scaled_reshaped = select_optimal_time_steps_and_features(X, y, config, db_name, timestamp)

        # Use SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled_reshaped.reshape(X_scaled_reshaped.shape[0], -1), y)
        X_resampled = X_resampled.reshape(-1, X_scaled_reshaped.shape[1], X_scaled_reshaped.shape[2])

        # Cross-validate and train the CNN-LSTM model
        cross_validate_cnn_lstm(X_resampled, y_resampled, config, timestamp, db_name)

