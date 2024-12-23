import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score

# Setup logging
log_file_path = '../../../logs/model_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])

# CNN model creation function with EarlyStopping
def create_cnn_model(input_shape, learning_rate=0.001, dropout_rate=0.2, filters=64, kernel_size=1):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())  # Flatten the feature maps before the Dense layer
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    logging.info(f"CNN model created with {filters} filters, kernel size {kernel_size}, and learning rate {learning_rate}")
    return model

# Function to evaluate model performance and print out confusion matrix, F1-Score, and precision-recall curve
def evaluate_model_performance(y_true, y_pred, db_name, fold):
    # Confusion Matrix and F1-Score
    cm = confusion_matrix(y_true, np.round(y_pred))
    f1 = f1_score(y_true, np.round(y_pred))

    logging.info(f"Confusion Matrix for {db_name}, Fold {fold}:\n{cm}")
    logging.info(f"F1-Score for {db_name}, Fold {fold}: {f1}")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    logging.info(f"Precision-Recall AUC for {db_name}, Fold {fold}: {pr_auc}")

    return f1, pr_auc

# Function to train CNN on a single fold with EarlyStopping
def train_cnn_on_fold(X_train, y_train, X_val, y_val, config, fold, model_dir, db_name):
    cnn_model = create_cnn_model((X_train.shape[1], X_train.shape[2]), learning_rate=config['learning_rate'],
                                 dropout_rate=config['dropout_rate'], filters=config['filters'], kernel_size=config['kernel_size'])

    # EarlyStopping callback to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history_cnn = cnn_model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                                validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Save model and history
    cnn_model_path = os.path.join(model_dir, f"{db_name}_fold_{fold}.keras")
    cnn_history_path = os.path.join(model_dir, f"{db_name}_fold_{fold}_history.csv")
    cnn_model.save(cnn_model_path)
    pd.DataFrame(history_cnn.history).to_csv(cnn_history_path)
    logging.info(f"CNN model and history saved for fold {fold} of {db_name}")

    # Predictions on the validation set
    y_pred_val = cnn_model.predict(X_val)

    # Evaluate model performance
    f1, pr_auc = evaluate_model_performance(y_val, y_pred_val, db_name, fold)

    return history_cnn, f1, pr_auc

# Function for cross-validation with additional performance tracking
def cross_validate_cnn(X, y, config, timestamp, db_name):
    skf = StratifiedKFold(n_splits=5)
    model_dir = os.path.join(config['model_save_path'], f'cnn_model_{db_name}')
    os.makedirs(model_dir, exist_ok=True)

    aucs = []
    f1_scores = []
    pr_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"Training on fold {fold}/5 for {db_name}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        history_cnn, f1, pr_auc = train_cnn_on_fold(X_train, y_train, X_val, y_val, config, fold, model_dir, db_name)
        auc = max(history_cnn.history['val_AUC'])
        aucs.append(auc)
        f1_scores.append(f1)
        pr_aucs.append(pr_auc)

    avg_auc = np.mean(aucs)
    avg_f1 = np.mean(f1_scores)
    avg_pr_auc = np.mean(pr_aucs)

    logging.info(f"Average AUC across folds for {db_name}: {avg_auc}")
    logging.info(f"Average F1-Score across folds for {db_name}: {avg_f1}")
    logging.info(f"Average Precision-Recall AUC across folds for {db_name}: {avg_pr_auc}")

    # Automatic adaptation if performance is poor or overfitting occurs
    if avg_auc < 0.7 or max(aucs) - min(aucs) > 0.1:  # Adjust based on thresholds
        logging.info(f"Low AUC detected or high variance between folds for {db_name}, adjusting hyperparameters...")
        config['learning_rate'] *= 0.5  # Reduce learning rate
        config['filters'] += 32  # Increase filters
        config['dropout_rate'] += 0.1  # Increase dropout rate
        logging.info(f"New parameters for {db_name} - Learning Rate: {config['learning_rate']}, Filters: {config['filters']}, Dropout Rate: {config['dropout_rate']}")
        cross_validate_cnn(X, y, config, timestamp, db_name)
    else:
        logging.info(f"Model performance for {db_name} is satisfactory.")

# Main script to train the CNN model on all three databases
if __name__ == "__main__":
    config = {
        'model_save_path': '../../../models',
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'filters': 64,
        'kernel_size': 1,
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
        'simulation': '../../../data/prepare_cnn_training_data/simulate_X_train.npy',
        'combined': '../../../data/prepare_cnn_training_data/combined_X_train.npy',
        'real': '../../../data/prepare_cnn_training_data/processed_X_train.npy'
    }

    for db_name, db_path in databases.items():
        logging.info(f"Processing database: {db_name}")
        X = np.load(db_path)
        y_path = db_path.replace('X_train', 'y_train')
        y = np.load(y_path)

        logging.info(f"Loaded X shape: {X.shape}")

        # Skip reshaping if already 3D (correct shape for CNN)
        if X.ndim == 2:
            # Standardize the data (2D case)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_reshaped = X_scaled.reshape(-1, X_scaled.shape[1], 1)  # Reshape to 3D for CNN
        elif X.ndim == 3:
            X_scaled_reshaped = X  # Already in 3D, skip reshaping

        # Balance the dataset
        # Balance the dataset
        data_balanced = pd.concat([pd.DataFrame(X_scaled_reshaped.reshape(X_scaled_reshaped.shape[0], -1)),
                                   pd.Series(y, name=config['target_column'])], axis=1)

        # Split into majority and minority classes
        majority = data_balanced[data_balanced[config['target_column']] == 0]
        minority = data_balanced[data_balanced[config['target_column']] == 1]

        # Resample minority class to balance the dataset
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

        # Combine the majority and upsampled minority class
        balanced_data = pd.concat([majority, minority_upsampled])

        # Split back into features (X) and target (y)
        X_balanced = balanced_data.iloc[:, :-1].values.reshape(-1, X_scaled_reshaped.shape[1],
                                                               X_scaled_reshaped.shape[2])
        y_balanced = balanced_data[config['target_column']].values

        # Cross-validate and adjust hyperparameters automatically
        cross_validate_cnn(X_balanced, y_balanced, config, timestamp, db_name)

