import os
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_and_save_isolation_forest_model(X_train, y_train, X_val, y_val, config, timestamp):
    isolation_forest_model_path = f"{config['isolation_forest_model_path']}_{timestamp}.joblib"

    if not os.path.exists(isolation_forest_model_path):
        isolation_forest_model = IsolationForest(contamination=0.1, random_state=42)
        isolation_forest_model.fit(X_train)
        joblib.dump(isolation_forest_model, isolation_forest_model_path)
        logging.info("Isolation Forest model trained and saved successfully")
    else:
        logging.info("Isolation Forest model already exists. Skipping training.")
