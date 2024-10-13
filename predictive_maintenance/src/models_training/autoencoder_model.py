import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def train_and_save_autoencoder_model(X_train, y_train, X_val, y_val, config, timestamp):
    autoencoder_model_path = f"{config['autoencoder_model_path']}_{timestamp}.keras"
    autoencoder_history_path = f"{config['autoencoder_history_path']}_{timestamp}.csv"

    if not os.path.exists(autoencoder_model_path):
        autoencoder_model = create_autoencoder_model((X_train.shape[1],))
        history_autoencoder = autoencoder_model.fit(X_train, X_train, epochs=config['epochs'],
                                                    batch_size=config['batch_size'], validation_data=(X_val, X_val))
        autoencoder_model.save(autoencoder_model_path)
        pd.DataFrame(history_autoencoder.history).to_csv(autoencoder_history_path)
        logging.info("Autoencoder model trained and saved successfully")
    else:
        logging.info("Autoencoder model already exists. Skipping training.")
