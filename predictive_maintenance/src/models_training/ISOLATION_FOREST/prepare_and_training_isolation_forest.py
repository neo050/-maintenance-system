import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
import matplotlib.pyplot as plt
import shap
import warnings
import joblib  # Import joblib for saving models

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up output directory
output_dir = '../../../models/anomaly_detection_model_combined/'
os.makedirs(output_dir, exist_ok=True)

# Modify the logging configuration
log_file_path = '../../../logs/anomaly_detection_training.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
logging.info("Starting supervised machine failure prediction process...")
logging.info("Loading dataset...")

Synthetic_Dataset = pd.read_csv("../../../data/simulation/synthetic_data.csv")
real_data = pd.read_csv("../../../data/raw/ai4i2020.csv")
df = pd.concat([real_data, Synthetic_Dataset], ignore_index=True)

# Drop irrelevant columns
df.drop(columns=['UDI', 'Product ID'], inplace=True)

# Encode categorical variables
if 'Type' in df.columns:
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])

# Feature Engineering based on failure conditions

# Map product types to thresholds
type_mapping = df[['Type']].drop_duplicates().reset_index(drop=True)
type_mapping['OSF_threshold'] = [11000, 12000, 13000]  # Adjust thresholds accordingly
df = df.merge(type_mapping, on='Type', how='left')

# Calculate additional features based on failure conditions
df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Rotational speed [rad/s]'] = df['Rotational speed [rpm]'] * (2 * np.pi / 60)
df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rad/s]']
df['Tool_Torque_Product'] = df['Tool wear [min]'] * df['Torque [Nm]']

# Failure condition features
df['TWF_condition'] = ((df['Tool wear [min]'] >= 200) & (df['Tool wear [min]'] <= 240)).astype(int)
df['HDF_condition'] = ((df['Temp_diff'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)).astype(int)
df['PWF_condition'] = ((df['Power'] < 3500) | (df['Power'] > 9000)).astype(int)
df['OSF_condition'] = (df['Tool_Torque_Product'] > df['OSF_threshold']).astype(int)

# Aggregate failure risk
df['Failure_Risk'] = (
    df['TWF_condition'] |
    df['HDF_condition'] |
    df['PWF_condition'] |
    df['OSF_condition']
).astype(int)

# Define features for selection
features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Type',
    'Temp_diff',
    'Rotational speed [rad/s]',
    'Power',
    'Tool_Torque_Product',
    'TWF_condition',
    'HDF_condition',
    'PWF_condition',
    'OSF_condition',
    'Failure_Risk'
]

# Separate features and target
X = df[features]
y = df['Machine failure']

# Address class imbalance by resampling
logging.info("Addressing class imbalance...")
df_majority = df[df["Machine failure"] == 0]
df_minority = df[df["Machine failure"] == 1]

# Upsample minority class
df_minority_upsampled = resample(
    df_minority,
    replace=True,  # Sample with replacement
    n_samples=len(df_majority),  # Match number of majority class
    random_state=42,
)

# Combine majority class with upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and target from balanced dataset
X_balanced = df_balanced[features]
y_balanced = df_balanced["Machine failure"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)
logging.info("Dataset split into training and testing sets.")

# Check for constant features
selector = VarianceThreshold(threshold=0.0)
selector.fit(X_train)
constant_features = [column for column in X_train.columns if column not in X_train.columns[selector.get_support()]]
if constant_features:
    logging.info(f"Removing constant features: {constant_features}")
    X_train.drop(columns=constant_features, inplace=True)
    X_test.drop(columns=constant_features, inplace=True)

# Ensure data does not contain NaN or infinite values
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.dropna(inplace=True)
X_test.dropna(inplace=True)
y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
scaler_filename = os.path.join(output_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_filename)
logging.info(f"Scaler saved to {scaler_filename}")

# Feature Selection using Random Forest for initial feature importance
logging.info("Performing feature selection using Random Forest...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train_scaled, y_train)

# Get feature importances and select features
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_train.columns

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()
logging.info("Feature importance plot saved to feature_importance.png")

# Select features with importance greater than a threshold
threshold = 0.01  # You can adjust this threshold
sfm = SelectFromModel(rf_selector, threshold=threshold, prefit=True)
selected_features = X_train.columns[sfm.get_support()]
logging.info(f"Selected features: {list(selected_features)}")

# Save the selected features
selected_features_filename = os.path.join(output_dir, 'selected_features.pkl')
joblib.dump(selected_features, selected_features_filename)
logging.info(f"Selected features saved to {selected_features_filename}")

# Update training and testing sets with selected features
X_train_selected = sfm.transform(X_train_scaled)
X_test_selected = sfm.transform(X_test_scaled)

# Hyperparameter tuning with Optuna for Random Forest, XGBoost, and LightGBM
def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["RandomForest", "XGBoost", "LightGBM"])
    if classifier_name == "RandomForest":
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 500)
        rf_max_depth = trial.suggest_int("rf_max_depth", 3, 20)
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=42,
            n_jobs=-1
        )
        model = rf
    elif classifier_name == "XGBoost":
        xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 100, 500)
        xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 20)
        xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)
        xgb = XGBClassifier(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        model = xgb
    else:
        lgb_n_estimators = trial.suggest_int("lgb_n_estimators", 100, 500)
        lgb_max_depth = trial.suggest_int("lgb_max_depth", 3, 17)  # Upper limit set to 17
        lgb_learning_rate = trial.suggest_float("lgb_learning_rate", 0.01, 0.3)
        max_num_leaves = min(2 ** lgb_max_depth, 131072)  # Ensure num_leaves <= 131072
        lgb_num_leaves = trial.suggest_int("lgb_num_leaves", 2, max_num_leaves)
        lgb_min_child_samples = trial.suggest_int("lgb_min_child_samples", 5, 30)
        lgb = LGBMClassifier(
            n_estimators=lgb_n_estimators,
            max_depth=lgb_max_depth,
            learning_rate=lgb_learning_rate,
            num_leaves=lgb_num_leaves,
            min_child_samples=lgb_min_child_samples,
            random_state=42,
            n_jobs=-1,
            verbose=-1  # Suppress LightGBM warnings
        )
        model = lgb

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        X_train_selected,
        y_train,
        cv=skf,
        scoring='f1',
        error_score='raise'
    )
    return scores.mean()

# Optimize hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
logging.info(f"Best hyperparameters: {study.best_params}")

# Save the Optuna study (optional)
study_filename = os.path.join(output_dir, 'optuna_study.pkl')
joblib.dump(study, study_filename)
logging.info(f"Optuna study saved to {study_filename}")

# Train the best model
if study.best_params["classifier"] == "RandomForest":
    best_model = RandomForestClassifier(
        n_estimators=study.best_params["rf_n_estimators"],
        max_depth=study.best_params["rf_max_depth"],
        random_state=42,
        n_jobs=-1
    )
elif study.best_params["classifier"] == "XGBoost":
    best_model = XGBClassifier(
        n_estimators=study.best_params["xgb_n_estimators"],
        max_depth=study.best_params["xgb_max_depth"],
        learning_rate=study.best_params["xgb_learning_rate"],
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
else:
    best_model = LGBMClassifier(
        n_estimators=study.best_params["lgb_n_estimators"],
        max_depth=study.best_params["lgb_max_depth"],
        learning_rate=study.best_params["lgb_learning_rate"],
        num_leaves=study.best_params["lgb_num_leaves"],
        min_child_samples=study.best_params["lgb_min_child_samples"],
        random_state=42,
        n_jobs=-1,
        verbose=-1  # Suppress LightGBM warnings
    )

# Fit the best model
best_model.fit(X_train_selected, y_train)

# Save the best model
best_model_filename = os.path.join(output_dir, 'best_model.pkl')
joblib.dump(best_model, best_model_filename)
logging.info(f"Best model saved to {best_model_filename}")

# Predictions and Evaluation
y_pred = best_model.predict(X_test_selected)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

logging.info(f"Best Model - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

# Classification report
report = classification_report(y_test, y_pred)
logging.info("Classification Report:")
logging.info(f"\n{report}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
logging.info(f"Confusion Matrix:\n{conf_matrix}")

# Feature importance plot for the best model
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = selected_features

plt.figure(figsize=(12, 6))
plt.title("Feature Importances of Best Model")
plt.bar(range(len(selected_features)), importances[indices], align='center')
plt.xticks(range(len(selected_features)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_model_feature_importance.png"))
plt.close()
logging.info("Best model feature importance plot saved to best_model_feature_importance.png")

# Model Interpretation using SHAP
logging.info("Generating SHAP values for model interpretability...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_selected)

# Adjust for LightGBM output format
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Plot SHAP summary plot
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test_selected, feature_names=selected_features, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
plt.close()
logging.info("SHAP summary plot saved to shap_summary_plot.png")

# Ensemble Model
logging.info("Creating an ensemble model using voting classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
lgb_model = LGBMClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=1024,  # Adjusted to ensure num_leaves <= 131072
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1  # Suppress LightGBM warnings
)

ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model), ('lgb', lgb_model)],
    voting='soft',
    n_jobs=-1
)

# Cross-validation
ensemble_scores = cross_val_score(ensemble, X_train_selected, y_train, cv=5, scoring='f1')
logging.info(f"Ensemble model cross-validation F1 score: {ensemble_scores.mean():.4f}")

# Fit ensemble model
ensemble.fit(X_train_selected, y_train)

# Save the ensemble model
ensemble_model_filename = os.path.join(output_dir, 'ensemble_model.pkl')
joblib.dump(ensemble, ensemble_model_filename)
logging.info(f"Ensemble model saved to {ensemble_model_filename}")

# Predictions and Evaluation
y_pred_ensemble = ensemble.predict(X_test_selected)
ensemble_precision = precision_score(y_test, y_pred_ensemble)
ensemble_recall = recall_score(y_test, y_pred_ensemble)
ensemble_f1 = f1_score(y_test, y_pred_ensemble)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

logging.info(f"Ensemble Model - Precision: {ensemble_precision:.4f}, Recall: {ensemble_recall:.4f}, F1 Score: {ensemble_f1:.4f}, Accuracy: {ensemble_accuracy:.4f}")

# Classification report for ensemble
ensemble_report = classification_report(y_test, y_pred_ensemble)
logging.info("Ensemble Model Classification Report:")
logging.info(f"\n{ensemble_report}")

# Confusion matrix for ensemble
ensemble_conf_matrix = confusion_matrix(y_test, y_pred_ensemble)
logging.info(f"Ensemble Model Confusion Matrix:\n{ensemble_conf_matrix}")

logging.info("Machine failure prediction process completed!")
