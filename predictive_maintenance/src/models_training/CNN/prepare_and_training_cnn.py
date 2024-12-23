import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import joblib

# Setup logging
log_file_path = '../../../logs/CNN_training.log'
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

# Check for class imbalance
logging.info(f'Class distribution before resampling: {target.value_counts()}')

# Normalize features
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)
# Save the scaler
scaler_filename = '../../../models/cnn_model_combined/scaler_nn.pkl'
joblib.dump(scaler, scaler_filename)
logging.info(f"Scaler saved to {scaler_filename}")
# Apply random oversampling to balance the class distribution
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(normalized_features, target)
logging.info(f'Class distribution after resampling: {pd.Series(y_resampled).value_counts()}')

# Prepare data for CNN by creating sequences
sequence_length = 5

def create_sequences(data, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
        labels.append(target[i + seq_length - 1])
    return np.array(sequences), np.array(labels)

# Apply k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_num = 1
all_histories = []  # To collect all histories for plotting later
for train_index, test_index in kf.split(X_resampled):
    logging.info(f"Training on Fold {fold_num}...")

    X_train_fold, X_test_fold = X_resampled[train_index], X_resampled[test_index]
    y_train_fold, y_test_fold = y_resampled[train_index], y_resampled[test_index]

    # Reset the indices of the target (y) data to avoid index mismatch after KFold splitting
    y_train_fold = pd.Series(y_train_fold).reset_index(drop=True)
    y_test_fold = pd.Series(y_test_fold).reset_index(drop=True)

    # Create sequences for CNN input
    X_train_seq, y_train_seq = create_sequences(X_train_fold, y_train_fold, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_fold, y_test_fold, sequence_length)

    # Reshape for CNN input
    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2]))
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2]))

    # Define CNN model
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(sequence_length, X_train_seq.shape[2]), padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Apply class weighting to prioritize recall during training
    class_weights = {0: 1, 1: 3}

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Train the model
    history = model.fit(X_train_seq, y_train_seq, class_weight=class_weights, epochs=50, batch_size=32,
                        validation_data=(X_test_seq, y_test_seq), callbacks=[early_stopping, reduce_lr])

    # Save the history for each fold
    all_histories.append(history.history)

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
    model_save_path = f"../../../models/cnn_model_combined/combined_fold_{fold_num}.keras"
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    # Save training history to a CSV file for each fold
    history_df = pd.DataFrame(history.history)
    history_save_path = f"../../../models/cnn_model_combined/history_fold_{fold_num}.csv"
    history_df.to_csv(history_save_path, index=False)
    logging.info(f"Training history saved to {history_save_path}")

    fold_num += 1

logging.info("K-Fold Cross-Validation Completed!")

# After all folds are done, plot the accuracy and loss across all folds
plt.figure(figsize=(12, 5))

# Plot accuracy values
plt.subplot(1, 2, 1)
for i, history in enumerate(all_histories):
    plt.plot(history['accuracy'], label=f'Fold {i + 1} Train Accuracy')
    plt.plot(history['val_accuracy'], label=f'Fold {i + 1} Validation Accuracy', linestyle='--')
plt.title('Model Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss values
plt.subplot(1, 2, 2)
for i, history in enumerate(all_histories):
    plt.plot(history['loss'], label=f'Fold {i + 1} Train Loss')
    plt.plot(history['val_loss'], label=f'Fold {i + 1} Validation Loss', linestyle='--')
plt.title('Model Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("../../../models/cnn_model_combined/plot_results.png")
plt.show()
