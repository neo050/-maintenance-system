import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import joblib

# Setup logging
log_file_path = '../../../logs/LSTM_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])

# Load datasets
Synthetic_Dataset = pd.read_csv("../../../data/simulation/synthetic_data.csv")
real_data = pd.read_csv("../../../data/raw/ai4i2020.csv")
combined_data = pd.concat([real_data, Synthetic_Dataset], ignore_index=True)

combined_data.to_csv("../../../data/combined/combined_data.csv", index=False)

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
scaler_filename = '../../../models/lstm_model_combined/scaler_nn.pkl'
joblib.dump(scaler, scaler_filename)
logging.info(f"Scaler saved to {scaler_filename}")


# Split data into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(normalized_features, target, test_size=0.2, random_state=42)

# Prepare data for LSTM by creating sequences
sequence_length = 5

def create_sequences(data, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
        labels.append(target[i + seq_length - 1])
    return np.array(sequences), np.array(labels)

X_train_seq_full, y_train_seq_full = create_sequences(X_train_full, y_train_full.values, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test.values, sequence_length)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_var = 1
all_histories = []

for train_index, val_index in kf.split(X_train_seq_full):
    logging.info(f"Training on Fold {fold_var}...")

    X_train_seq = X_train_seq_full[train_index]
    X_val_seq = X_train_seq_full[val_index]
    y_train_seq = y_train_seq_full[train_index]
    y_val_seq = y_train_seq_full[val_index]

    # Define and compile the LSTM model
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(sequence_length, X_train_seq.shape[2]),
               kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        BatchNormalization(),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model
    history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32,
                        validation_data=(X_val_seq, y_val_seq), callbacks=[early_stopping, reduce_lr])

    # Save the history
    all_histories.append(history.history)

    # Save model
    model.save(f'../../../models/lstm_model_combined/model_fold_{fold_var}.keras')

    # Logging training progress
    logging.info(f'Training fold {fold_var} completed.')

    # Evaluate model on the test set for this fold
    test_loss, test_accuracy = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    y_probs = model.predict(X_test_seq)

    # Use ROC curve to find the optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test_seq, y_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_probs > optimal_threshold).astype(int)

    precision = precision_score(y_test_seq, y_pred_optimal)
    recall = recall_score(y_test_seq, y_pred_optimal)
    f1 = f1_score(y_test_seq, y_pred_optimal)
    roc_auc = roc_auc_score(y_test_seq, y_probs)

    # Save history with additional test metrics
    hist_df = pd.DataFrame(history.history)
    hist_df['test_loss'] = test_loss
    hist_df['test_accuracy'] = test_accuracy
    hist_df['precision'] = precision
    hist_df['recall'] = recall
    hist_df['f1_score'] = f1
    hist_df['roc_auc'] = roc_auc
    hist_df.to_csv(f'../../../models/lstm_model_combined/history_fold_{fold_var}.csv')

    # Logging evaluation results
    logging.info(f'Test metrics for fold {fold_var}: Loss={test_loss}, Accuracy={test_accuracy}, Precision={precision}, Recall={recall}, F1 Score={f1}, ROC-AUC={roc_auc}')

    fold_var += 1

logging.info("K-Fold Cross-Validation Completed!")

# Plot the results
plt.figure(figsize=(12, 10))

# Plot accuracy and loss for each fold
for i, history in enumerate(all_histories):
    # Accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history['accuracy'], label=f'Fold {i+1} Training Accuracy')
    plt.plot(history['val_accuracy'], label=f'Fold {i+1} Validation Accuracy', linestyle='--')

    # Loss
    plt.subplot(2, 1, 2)
    plt.plot(history['loss'], label=f'Fold {i+1} Training Loss')
    plt.plot(history['val_loss'], label=f'Fold {i+1} Validation Loss', linestyle='--')

plt.subplot(2, 1, 1)
plt.title('Model Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Model Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("../../../models/lstm_model_combined/plot_results.png")
plt.show()
