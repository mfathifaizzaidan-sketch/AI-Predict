from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model at startup
model_path = os.path.join(os.path.dirname(__file__), 'model_prediksi_saldo_mingguan.pkl')
model = None

def load_model():
    global model
    if model is None:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    return model

# Load model on startup
load_model()

@app.route('/', methods=['GET'])
def home():
    """Root endpoint - API info"""
    return jsonify({
        "success": True,
        "message": "AI Prediction API untuk Bank Sampah OSKU",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/predict",
            "health": "/health"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/api/predict', methods=['GET', 'OPTIONS'])
@app.route('/predict', methods=['GET', 'OPTIONS'])
def predict_info():
    """GET - return API info"""
    return jsonify({
        "success": True,
        "message": "AI Prediction API untuk Bank Sampah OSKU",
        "version": "1.0.0",
        "endpoint": "/api/predict",
        "method": "POST",
        "expected_features": [
            "lag_1",
            "lag_2",
            "lag_3",
            "ma_4",
            "bulan",
            "minggu_dalam_bulan"
        ],
        "example_request": {
            "features": {
                "lag_1": 1000000,
                "lag_2": 950000,
                "lag_3": 900000,
                "ma_4": 950000,
                "bulan": 12,
                "minggu_dalam_bulan": 4
            }
        }
    })

@app.route('/api/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
@app.route('/', methods=['POST'])
def predict():
    """POST - make prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Extract features - support both nested and flat format
        features = data.get('features', data)
        
        # Validate required features
        required_features = ['lag_1', 'lag_2', 'lag_3', 'ma_4', 'bulan', 'minggu_dalam_bulan']
        missing_features = [f for f in required_features if f not in features]
        
        if missing_features:
            return jsonify({
                "success": False,
                "error": f"Missing required features: {missing_features}",
                "expected_features": required_features,
                "received": list(features.keys()) if isinstance(features, dict) else []
            }), 400
        
        # Prepare features array for prediction
        # Order: lag_1, lag_2, lag_3, ma_4, bulan, minggu_dalam_bulan
        feature_values = [
            float(features['lag_1']),
            float(features['lag_2']),
            float(features['lag_3']),
            float(features['ma_4']),
            float(features['bulan']),
            float(features['minggu_dalam_bulan'])
        ]
        
        # Make prediction
        X = np.array([feature_values])
        prediction = model.predict(X)[0]
        
        return jsonify({
            "success": True,
            "prediction": float(prediction),
            "message": "Prediksi saldo mingguan berhasil",
            "input_features": features
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid feature value: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
