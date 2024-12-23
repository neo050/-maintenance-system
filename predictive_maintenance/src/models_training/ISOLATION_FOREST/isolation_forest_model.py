import os
import pandas as pd
import numpy as np
import logging

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef,
                             precision_recall_curve, auc)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib

# Suppress warnings to keep the output clean
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_file_path = '../../../logs/anomaly_detection_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])

# Metrics to store for each fold
history_columns = ['Fold', 'AUC', 'Average_Precision', 'Precision', 'Recall',
                   'F1_Score', 'Accuracy', 'Loss', 'Training_Accuracy',
                   'Validation_Accuracy', 'Training_Loss', 'Validation_Loss',
                   'Matthews_Corrcoef', 'PR_AUC', 'Contamination_Level', 'Model_Name']

def safe_auc_score(y_true, y_scores):
    """
    Calculate AUC score safely.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_scores)

def safe_pr_auc_score(y_true, y_scores):
    """
    Calculate Precision-Recall AUC safely.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def scale_and_reduce(X_train, X_val):
    """
    Scale the data using StandardScaler and reduce dimensionality using PCA.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Limit the number of components to reduce computational load
    pca = PCA(n_components=20, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    return X_train_pca, X_val_pca

def create_lof_model(X_train, contamination):
    """
    Create and fit a Local Outlier Factor (LOF) model.
    """
    model = LocalOutlierFactor(novelty=True, n_neighbors=20, contamination=contamination)
    model.fit(X_train)
    return model

def create_ocsvm_model(X_train, contamination):
    """
    Create and fit a One-Class SVM model.
    """
    model = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
    model.fit(X_train)
    return model

def create_dbscan_model(X_train):
    """
    Create and fit a DBSCAN model.
    """
    # Using default parameters as DBSCAN is sensitive to eps and min_samples
    model = DBSCAN(eps=0.5, min_samples=5, n_jobs=4)
    model.fit(X_train)
    return model

def create_isolation_forest_model(X_train, contamination, random_state=42):
    """
    Create and fit an Isolation Forest model.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        max_samples='auto',
        n_jobs=4,
        random_state=random_state
    )
    model.fit(X_train)
    return model

def create_random_forest_model(X_train, y_train, random_state=42):
    """
    Create and fit a Random Forest Classifier.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',  # Changed from 'auto' to 'sqrt'
        class_weight='balanced',
        n_jobs=4,
        random_state=random_state
    )
    model.fit(X_resampled, y_resampled)
    return model

def dynamic_weight_adjustment(models, X_val, y_val):
    """
    Adjust model weights based on validation performance.
    """
    model_weights = {}
    total_f1 = 0
    for model_name, model in models.items():
        if model_name != 'DBSCAN':
            if model_name in ['IsolationForest', 'OneClassSVM', 'LOF']:
                y_scores = -model.decision_function(X_val)
            else:
                y_scores = model.predict_proba(X_val)[:, 1]
            y_pred = np.where(y_scores > 0.5, 1, 0)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            model_weights[model_name] = f1
            total_f1 += f1

    # Normalize weights
    if total_f1 == 0:
        # Assign equal weights if all f1 scores are zero
        for model_name in model_weights:
            model_weights[model_name] = 1 / len(model_weights)
    else:
        for model_name in model_weights:
            model_weights[model_name] /= total_f1

    return model_weights

def weighted_majority_voting(models, X, weights):
    """
    Combine model predictions using weighted majority voting.
    """
    weighted_preds = np.zeros(X.shape[0])
    total_weight = 0
    for model_name, model in models.items():
        weight = weights.get(model_name, 0)
        if weight > 0:
            if model_name != 'DBSCAN':
                if model_name in ['IsolationForest', 'OneClassSVM', 'LOF']:
                    y_scores = -model.decision_function(X)
                else:
                    y_scores = model.predict_proba(X)[:, 1]
                weighted_preds += y_scores * weight
                total_weight += weight
    if total_weight == 0:
        total_weight = 1  # Prevent division by zero
    final_predictions = np.where(weighted_preds / total_weight > 0.5, 1, 0)
    return final_predictions

def evaluate_model(y_true, y_pred, y_scores, model_name):
    """
    Evaluate model performance and return metrics.
    """
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    auc_score = safe_auc_score(y_true, y_scores)
    pr_auc = safe_pr_auc_score(y_true, y_scores)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    loss = 1 - f1  # Approximate loss as 1 - F1 score

    # Logging metrics
    logging.info(f"Evaluation for {model_name}:")
    logging.info(f"Precision: {precision}, Recall: {recall}, F1: {f1}, "
                 f"Accuracy: {accuracy}, Average Precision: {avg_precision}, AUC: {auc_score}, PR AUC: {pr_auc}, MCC: {mcc}")

    return {
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Accuracy': accuracy,
        'Average_Precision': avg_precision,
        'AUC': auc_score,
        'PR_AUC': pr_auc,
        'Matthews_Corrcoef': mcc,
        'Loss': loss
    }

def train_multiple_models_and_vote(X_train, X_val, y_train, y_val, config, fold, model_dir, db_name, contamination):
    """
    Train multiple models, perform weighted voting, and evaluate.
    """
    X_train_scaled, X_val_scaled = scale_and_reduce(X_train, X_val)

    # Models dictionary
    models = {
        'IsolationForest': create_isolation_forest_model(X_train_scaled, contamination),
        'LOF': create_lof_model(X_train_scaled, contamination),
        'OneClassSVM': create_ocsvm_model(X_train_scaled, contamination),
        'RandomForest': create_random_forest_model(X_train_scaled, y_train),
        'DBSCAN': create_dbscan_model(X_train_scaled)
    }

    # Save models and collect metrics
    model_metrics = {}
    for model_name, model in models.items():
        if model_name == 'DBSCAN':
            # For DBSCAN, labels are obtained directly
            y_val_pred = model.fit_predict(X_val_scaled)
            y_val_pred_binary = np.where(y_val_pred == -1, 1, 0)
            y_train_pred = model.fit_predict(X_train_scaled)
            y_train_pred_binary = np.where(y_train_pred == -1, 1, 0)
            y_val_scores = np.full_like(y_val_pred_binary, 0.5)
            y_train_scores = np.full_like(y_train_pred_binary, 0.5)
        elif model_name in ['IsolationForest', 'OneClassSVM', 'LOF']:
            y_val_scores = -model.decision_function(X_val_scaled)
            y_val_pred_binary = np.where(y_val_scores > 0.5, 1, 0)
            y_train_scores = -model.decision_function(X_train_scaled)
            y_train_pred_binary = np.where(y_train_scores > 0.5, 1, 0)
        else:
            y_val_scores = model.predict_proba(X_val_scaled)[:, 1]
            y_val_pred_binary = np.where(y_val_scores > 0.5, 1, 0)
            y_train_scores = model.predict_proba(X_train_scaled)[:, 1]
            y_train_pred_binary = np.where(y_train_scores > 0.5, 1, 0)

        # Evaluate model
        val_metrics = evaluate_model(y_val, y_val_pred_binary, y_val_scores, model_name)
        train_metrics = evaluate_model(y_train, y_train_pred_binary, y_train_scores, model_name + " (Train)")
        model_metrics[model_name] = {'Validation': val_metrics, 'Training': train_metrics}

        # Save the model
        model_path = os.path.join(model_dir, f"{model_name}_fold_{fold}_contamination_{contamination}.joblib")
        joblib.dump(model, model_path)
        logging.info(f"Saved model {model_name} to {model_path}")

    # Dynamic weight adjustment and voting on validation data
    model_weights = dynamic_weight_adjustment(models, X_val_scaled, y_val)
    final_predictions = weighted_majority_voting(models, X_val_scaled, weights=model_weights)

    # Combine all training predictions for majority voting
    final_train_predictions = weighted_majority_voting(models, X_train_scaled, weights=model_weights)

    # Evaluate ensemble
    val_metrics = evaluate_model(y_val, final_predictions, final_predictions, "Ensemble")
    train_metrics = evaluate_model(y_train, final_train_predictions, final_train_predictions, "Ensemble (Train)")
    model_metrics['Ensemble'] = {'Validation': val_metrics, 'Training': train_metrics}

    return model_metrics

def cross_validate(X, y, config, db_name, contamination):
    """
    Perform cross-validation and model training.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model_dir = os.path.join(config['model_save_path'], f'anomaly_detection_model_{db_name}')
    os.makedirs(model_dir, exist_ok=True)

    fold_histories = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"Training fold {fold}/3 for {db_name} with contamination {contamination}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_metrics = train_multiple_models_and_vote(
            X_train, X_val, y_train, y_val, config, fold, model_dir, db_name, contamination)

        # Save history for each model
        for model_name, metrics in model_metrics.items():
            history_data = pd.DataFrame({
                'Fold': [fold],
                'AUC': [metrics['Validation']['AUC']],
                'Average_Precision': [metrics['Validation']['Average_Precision']],
                'Precision': [metrics['Validation']['Precision']],
                'Recall': [metrics['Validation']['Recall']],
                'F1_Score': [metrics['Validation']['F1_Score']],
                'Accuracy': [metrics['Validation']['Accuracy']],
                'Loss': [metrics['Validation']['Loss']],
                'Training_Accuracy': [metrics['Training']['Accuracy']],
                'Validation_Accuracy': [metrics['Validation']['Accuracy']],
                'Training_Loss': [metrics['Training']['Loss']],
                'Validation_Loss': [metrics['Validation']['Loss']],
                'Matthews_Corrcoef': [metrics['Validation']['Matthews_Corrcoef']],
                'PR_AUC': [metrics['Validation']['PR_AUC']],
                'Contamination_Level': [contamination],
                'Model_Name': [model_name]
            })
            fold_histories.append(history_data)
            logging.info(f"Metrics for model {model_name} on fold {fold} collected.")

    # Combine histories and save
    full_history = pd.concat(fold_histories, ignore_index=True)
    history_file = os.path.join(model_dir, f"history_contamination_{contamination}.csv")
    full_history.to_csv(history_file, index=False)
    logging.info(f"Saved full training history to {history_file}")

if __name__ == "__main__":
    config = {
        'model_save_path': '../../../models',
    }

    contamination_levels = [0.005, 0.01, 0.02]
    databases = {
        'simulation': '../../../data/prepare_anomaly_training_data/simulation_X_train.csv',
        'combined': '../../../data/prepare_anomaly_training_data/combined_X_train.csv',
        'real': '../../../data/prepare_anomaly_training_data/processed_X_train.csv'
    }

    for db_name, db_path in databases.items():
        logging.info(f"Processing database: {db_name}")

        # Use generators to read data in chunks to optimize memory usage
        X_data_iter = pd.read_csv(db_path, chunksize=10000, dtype=np.float32)
        X_data = pd.concat([chunk for chunk in X_data_iter], ignore_index=True)

        y_data_iter = pd.read_csv(db_path.replace('X_train.csv', 'y_train.csv'), chunksize=10000, dtype=np.float32)
        y_data = pd.concat([chunk for chunk in y_data_iter], ignore_index=True)

        X = X_data.to_numpy(dtype=np.float32)
        y = y_data.to_numpy(dtype=np.float32).ravel()

        logging.info(f"Loaded X shape: {X.shape}, Loaded y shape: {y.shape}")

        for contamination in contamination_levels:
            logging.info(f"Running with contamination level: {contamination}")
            cross_validate(X, y, config, db_name, contamination)
