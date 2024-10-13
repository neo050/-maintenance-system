import logging
import os
import joblib
from tensorflow.keras.models import load_model


models = {
    'cnn': {'models': [], 'scaler': None},
    'cnn_lstm': {'models': [], 'scaler': None},
    'lstm': {'models': [], 'scaler': None},
    'supervised': {'models': [], 'scaler': None},
}
# Load models
def load_models(models_dir):
    # CNN models
    cnn_model_dir = os.path.join(models_dir, 'cnn_model_combined')
    cnn_model_files = [
        os.path.join(cnn_model_dir, file)
        for file in os.listdir(cnn_model_dir) if file.endswith('.keras')
    ]
    # Load scaler for CNN models
    scaler_file = os.path.join(cnn_model_dir, 'scaler_nn.pkl')
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        print(f"Scaler loaded for CNN models: {scaler_file}")
    else:
        print(f"Scaler file not found for CNN models: {scaler_file}")
        scaler = None
    # Load CNN models and associate the scaler
    models['cnn']['models'] = [load_model(file) for file in cnn_model_files]
    models['cnn']['scaler'] = scaler  # Single scaler for all CNN models
    print(f"{len(models['cnn']['models'])} CNN models loaded.")

    # CNN-LSTM models
    cnn_lstm_model_dir = os.path.join(models_dir, 'cnn_lstm_model_combined')
    cnn_lstm_model_files = [
        os.path.join(cnn_lstm_model_dir, file)
        for file in os.listdir(cnn_lstm_model_dir) if file.endswith('.keras')
    ]
    # Load scaler for CNN-LSTM models
    scaler_file = os.path.join(cnn_lstm_model_dir, 'scaler_nn.pkl')
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        print(f"Scaler loaded for CNN-LSTM models: {scaler_file}")
    else:
        print(f"Scaler file not found for CNN-LSTM models: {scaler_file}")
        scaler = None
    # Load CNN-LSTM models and associate the scaler
    models['cnn_lstm']['models'] = [load_model(file) for file in cnn_lstm_model_files]
    models['cnn_lstm']['scaler'] = scaler
    print(f"{len(models['cnn_lstm']['models'])} CNN-LSTM models loaded.")

    # LSTM models
    lstm_model_dir = os.path.join(models_dir, 'lstm_model_combined')
    lstm_model_files = [
        os.path.join(lstm_model_dir, file)
        for file in os.listdir(lstm_model_dir) if file.endswith('.keras')
    ]
    # Load scaler for LSTM models
    scaler_file = os.path.join(lstm_model_dir, 'scaler_nn.pkl')
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        print(f"Scaler loaded for LSTM models: {scaler_file}")
    else:
        print(f"Scaler file not found for LSTM models: {scaler_file}")
        scaler = None
    # Load LSTM models and associate the scaler
    models['lstm']['models'] = [load_model(file) for file in lstm_model_files]
    models['lstm']['scaler'] = scaler
    print(f"{len(models['lstm']['models'])} LSTM models loaded.")

    # Supervised models
    supervised_model_dir = os.path.join(models_dir, 'anomaly_detection_model_combined')
    supervised_model_files = [
        os.path.join(supervised_model_dir, file)
        for file in os.listdir(supervised_model_dir) if file.endswith('best_model.pkl') and 'model' in file
    ]
    # Load scaler for supervised models
    scaler_file = os.path.join(supervised_model_dir, 'scaler.pkl')
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        print(f"Scaler loaded for supervised models: {scaler_file}")
    else:
        print(f"Scaler file not found for supervised models: {scaler_file}")
        scaler = None
    # Load supervised models and associate the scaler
    models['supervised']['models'] = [joblib.load(file) for file in supervised_model_files]
    models['supervised']['scaler'] = scaler
    print(f"{len(models['supervised']['models'])} supervised models loaded.")

    print("All models and scalers loaded successfully.")







models_dir = '../models'


load_models(models_dir)

for i in models.items():
    if hasattr(i[1]['scaler'], 'feature_names_in_'):
        scaler_feature_names =i[1]['scaler'].feature_names_in_
        print(f"{i[0]}: Scaler feature names: {scaler_feature_names}")
    else:
       print("Scaler does not have feature names attribute.")