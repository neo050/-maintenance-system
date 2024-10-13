import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, TimeDistributed, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import logging
import matplotlib.pyplot as plt
import json
import joblib


# Setup logging
log_file_path = '../../../logs/CNN_LSTM_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])

# Load datasets
Synthetic_Dataset = pd.read_csv("../../../data/simulation/synthetic_data.csv")
real_data = pd.read_csv("../../../data/raw/ai4i2020.csv")
combined_data = pd.concat([real_data, Synthetic_Dataset], ignore_index=True)

# Feature Engineering Based on Failure Modes
thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
combined_data['OSF_threshold'] = combined_data['Type'].map(thresholds)

# Calculate additional features
combined_data['Temp_diff'] = combined_data['Process temperature [K]'] - combined_data['Air temperature [K]']
combined_data['Rotational speed [rad/s]'] = combined_data['Rotational speed [rpm]'] * (2 * np.pi / 60)
combined_data['Power'] = combined_data['Torque [Nm]'] * combined_data['Rotational speed [rad/s]']
combined_data['Tool_Torque_Product'] = combined_data['Tool wear [min]'] * combined_data['Torque [Nm]']

# Failure condition features
combined_data['TWF_condition'] = ((combined_data['Tool wear [min]'] >= 200) & (combined_data['Tool wear [min]'] <= 240)).astype(int)
combined_data['HDF_condition'] = ((combined_data['Temp_diff'] < 8.6) & (combined_data['Rotational speed [rpm]'] < 1380)).astype(int)
combined_data['PWF_condition'] = ((combined_data['Power'] < 3500) | (combined_data['Power'] > 9000)).astype(int)
combined_data['OSF_condition'] = (combined_data['Tool_Torque_Product'] > combined_data['OSF_threshold']).astype(int)

# Aggregate failure risk
combined_data['Failure_Risk'] = (
    combined_data['TWF_condition'] |
    combined_data['HDF_condition'] |
    combined_data['PWF_condition'] |
    combined_data['OSF_condition']
).astype(int)

# Define features and target
feature_columns = [
    'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
    'Torque [Nm]', 'Tool wear [min]', 'Temp_diff', 'Rotational speed [rad/s]',
    'Power', 'Tool_Torque_Product', 'TWF_condition', 'HDF_condition',
    'PWF_condition', 'OSF_condition', 'Failure_Risk'
]
features = combined_data[feature_columns]
target = combined_data['Machine failure']

# Normalize features
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# Save the scaler
scaler_filename = '../../../models/cnn_lstm_model_combined/scaler_nn.pkl'
joblib.dump(scaler, scaler_filename)
logging.info(f"Scaler saved to {scaler_filename}")

# Apply random oversampling to balance the class distribution
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(normalized_features, target)

# Prepare data for CNN-LSTM by creating sequences
sequence_length = 5

def create_sequences(data, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
        labels.append(target.iloc[i + seq_length - 1])  # Use .iloc for positional indexing
    return np.array(sequences), np.array(labels)

# Apply k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_num = 1
all_histories = []  # To collect all histories for plotting later
for train_index, test_index in kf.split(X_resampled):
    logging.info(f"Training on Fold {fold_num}...")

    X_train_fold, X_test_fold = X_resampled[train_index], X_resampled[test_index]
    y_train_fold, y_test_fold = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    # Create sequences for CNN-LSTM input
    X_train_seq, y_train_seq = create_sequences(X_train_fold, y_train_fold.reset_index(drop=True), sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_fold, y_test_fold.reset_index(drop=True), sequence_length)

    # Reshape for CNN-LSTM input
    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], sequence_length, X_train_seq.shape[2], 1))
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], sequence_length, X_test_seq.shape[2], 1))

    # Define CNN-LSTM model with L2 regularization and Dropout
    model = Sequential([
        # CNN Layers for feature extraction
        TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(0.001)),
                        input_shape=(sequence_length, X_train_seq.shape[2], 1)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Dropout(0.3)),

        TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(0.001))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Dropout(0.3)),

        # Flattening CNN output to feed into LSTM
        TimeDistributed(Flatten()),

        # LSTM Layers for temporal patterns
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),

        # Fully connected layer with L2 regularization
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),

        # Output layer for binary classification
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Class weights to handle class imbalance
    class_weights = {0: 1, 1: 3}

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Train the model
    history = model.fit(X_train_seq, y_train_seq, class_weight=class_weights, epochs=50, batch_size=32,
                        validation_data=(X_test_seq, y_test_seq), callbacks=[early_stopping, reduce_lr])

    # Save the history for each fold
    all_histories.append(history.history)

    # Save history to file for each fold (in JSON format)
    history_save_path = f"../../../models/cnn_lstm_model_combined/cnn_lstm_fold_{fold_num}_history.json"
    with open(history_save_path, 'w') as f:
        json.dump(history.history, f)
    logging.info(f"History saved to {history_save_path}")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test_seq, y_test_seq)
    y_probs = model.predict(X_test_seq)

    # Use ROC curve to find the optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test_seq, y_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_probs > optimal_threshold).astype(int)

    # Calculate metrics
    precision_optimal = precision_score(y_test_seq, y_pred_optimal)
    recall_optimal = recall_score(y_test_seq, y_pred_optimal)
    f1_optimal = f1_score(y_test_seq, y_pred_optimal)
    roc_auc = roc_auc_score(y_test_seq, y_probs)

    # Log metrics for each fold
    logging.info(f"Fold {fold_num} - Test Loss: {test_loss}")
    logging.info(f"Fold {fold_num} - Test Accuracy: {test_accuracy}")
    logging.info(f"Fold {fold_num} - Optimal Threshold: {optimal_threshold}")
    logging.info(f"Fold {fold_num} - Precision (Optimal Threshold): {precision_optimal}")
    logging.info(f"Fold {fold_num} - Recall (Optimal Threshold): {recall_optimal}")
    logging.info(f"Fold {fold_num} - F1 Score (Optimal Threshold): {f1_optimal}")
    logging.info(f"Fold {fold_num} - ROC-AUC: {roc_auc}")

    # Save model for each fold
    model_save_path = f"../../../models/cnn_lstm_model_combined/cnn_lstm_fold_{fold_num}.keras"
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    fold_num += 1

logging.info("K-Fold Cross-Validation Completed!")

# After all folds have been trained, we can plot the history for each fold

# Initialize lists for collecting data for plotting
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

# Collect metrics for each fold
for history in all_histories:
    train_accuracies.append(history['accuracy'])
    val_accuracies.append(history['val_accuracy'])
    train_losses.append(history['loss'])
    val_losses.append(history['val_loss'])

# Create subplots for accuracy and loss in a single image
plt.figure(figsize=(14, 10))

# Subplot 1: Training and Validation Accuracy
plt.subplot(2, 1, 1)
for i in range(len(train_accuracies)):
    plt.plot(train_accuracies[i], label=f'Training Accuracy - Fold {i+1}')
    plt.plot(val_accuracies[i], label=f'Validation Accuracy - Fold {i+1}', linestyle='dashed')
plt.title('Training and Validation Accuracy per Fold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Subplot 2: Training and Validation Loss
plt.subplot(2, 1, 2)
for i in range(len(train_losses)):
    plt.plot(train_losses[i], label=f'Training Loss - Fold {i + 1}')
    plt.plot(val_losses[i], label=f'Validation Loss - Fold {i + 1}', linestyle='dashed')
plt.title('Training and Validation Loss per Fold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the combined figure as one image
plt.tight_layout()
plt.savefig("../../../models/cnn_lstm_model_combined/training_validation_accuracy_loss_per_fold.png")
plt.show()
