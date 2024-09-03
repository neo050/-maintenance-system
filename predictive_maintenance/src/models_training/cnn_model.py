import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        logging.info("CNN model trained and saved successfully")
    else:
        logging.info("CNN model already exists. Skipping training.")
