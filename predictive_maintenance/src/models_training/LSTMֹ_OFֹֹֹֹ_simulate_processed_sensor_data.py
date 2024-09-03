import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('../../data/simulation/simulate_processed_sensor_data_with_lags.csv')

# Feature and target columns
feature_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                   'Tool wear [min]']
target_column = 'Machine failure'

# Extract features and target
X = data[feature_columns].values
y = data[target_column].values

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a function to create the LSTM model
def create_lstm_model(lstm_units=50, dropout_rate=0.2, learning_rate=0.001, l2_reg=0.01):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Implement a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(scheduler)

# Define the hyperparameters to tune
param_grid = {
    'lstm_units': [50, 100],
    'dropout_rate': [0.2, 0.3, 0.5],
    'learning_rate': [0.001, 0.0005],
    'l2_reg': [0.01, 0.001],
    'batch_size': [32, 64],
    'epochs': [20, 30]
}


# Function to wrap your model for GridSearchCV
def build_model(lstm_units, dropout_rate, learning_rate, l2_reg):
    return create_lstm_model(
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        l2_reg=l2_reg
    )


# Implement cross-validation
best_score = 0
best_params = None
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for lstm_units in param_grid['lstm_units']:
    for dropout_rate in param_grid['dropout_rate']:
        for learning_rate in param_grid['learning_rate']:
            for l2_reg in param_grid['l2_reg']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:

                        fold_aucs = []
                        fold_precisions = []
                        fold_recalls = []

                        for train_index, val_index in kf.split(X_train):
                            X_fold_train, X_val = X_train[train_index], X_train[val_index]
                            y_fold_train, y_val = y_train[train_index], y_train[val_index]

                            model = build_model(lstm_units, dropout_rate, learning_rate, l2_reg)

                            history = model.fit(
                                X_fold_train, y_fold_train,
                                validation_data=(X_val, y_val),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[early_stopping, lr_scheduler],
                                verbose=0
                            )

                            val_preds = model.predict(X_val)
                            val_auc = roc_auc_score(y_val, val_preds)
                            val_precision = precision_score(y_val, (val_preds > 0.5).astype(int))
                            val_recall = recall_score(y_val, (val_preds > 0.5).astype(int))

                            fold_aucs.append(val_auc)
                            fold_precisions.append(val_precision)
                            fold_recalls.append(val_recall)

                        avg_auc = np.mean(fold_aucs)
                        avg_precision = np.mean(fold_precisions)
                        avg_recall = np.mean(fold_recalls)

                        print(f"Params: LSTM Units: {lstm_units}, Dropout Rate: {dropout_rate}, "
                              f"Learning Rate: {learning_rate}, L2 Reg: {l2_reg}, Batch Size: {batch_size}, Epochs: {epochs}")
                        print(f"- AUC: {avg_auc:.4f} - Precision: {avg_precision:.4f} - Recall: {avg_recall:.4f}")

                        if avg_auc > best_score:
                            best_score = avg_auc
                            best_params = {
                                'lstm_units': lstm_units,
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'l2_reg': l2_reg,
                                'batch_size': batch_size,
                                'epochs': epochs
                            }

print(f"Best Parameters: {best_params}")

# Retrain the model with best hyperparameters
best_model = build_model(**best_params)
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Evaluate on test data
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
test_auc = roc_auc_score(y_test, best_model.predict(X_test))
test_precision = precision_score(y_test, (best_model.predict(X_test) > 0.5).astype(int))
test_recall = recall_score(y_test, (best_model.predict(X_test) > 0.5).astype(int))

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Save the model
best_model.save("best_model.keras")
