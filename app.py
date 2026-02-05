from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from datetime import datetime

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

def calculate_features(data, weeks=1):
    """
    Calculate lag features from time series data
    data: list of weekly balance values (most recent last)
    Returns: features dict with lag_1, lag_2, lag_3, ma_4, bulan, minggu_dalam_bulan
    """
    if len(data) < 4:
        raise ValueError("Need at least 4 weeks of data to calculate features")
    
    # Get the last 4 values for lag features
    values = [float(v) for v in data[-4:]]
    
    # lag_1 = most recent value
    # lag_2 = second most recent
    # lag_3 = third most recent
    lag_1 = values[-1]
    lag_2 = values[-2]
    lag_3 = values[-3]
    
    # ma_4 = 4-week moving average
    ma_4 = sum(values) / 4
    
    # Current date for bulan and minggu_dalam_bulan
    now = datetime.now()
    bulan = now.month
    minggu_dalam_bulan = min((now.day - 1) // 7 + 1, 5)
    
    return {
        'lag_1': lag_1,
        'lag_2': lag_2,
        'lag_3': lag_3,
        'ma_4': ma_4,
        'bulan': bulan,
        'minggu_dalam_bulan': minggu_dalam_bulan
    }

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
        "supported_formats": [
            {
                "format": "raw_data",
                "description": "Send raw weekly data, API calculates features",
                "example": {
                    "data": [1000000, 1100000, 1050000, 1200000],
                    "weeks": 1
                }
            },
            {
                "format": "features",
                "description": "Send pre-calculated features",
                "example": {
                    "features": {
                        "lag_1": 1200000,
                        "lag_2": 1050000,
                        "lag_3": 1100000,
                        "ma_4": 1087500,
                        "bulan": 12,
                        "minggu_dalam_bulan": 4
                    }
                }
            }
        ]
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
        
        # Check if using raw data format or features format
        if 'data' in data:
            # Raw data format - calculate features from time series
            raw_data = data.get('data', [])
            weeks = data.get('weeks', 1)
            
            if not raw_data or len(raw_data) < 4:
                return jsonify({
                    "success": False,
                    "error": "Need at least 4 weeks of data in 'data' array"
                }), 400
            
            features = calculate_features(raw_data, weeks)
            
        elif 'features' in data:
            # Pre-calculated features format
            features = data.get('features', {})
            
            # Validate required features
            required_features = ['lag_1', 'lag_2', 'lag_3', 'ma_4', 'bulan', 'minggu_dalam_bulan']
            missing_features = [f for f in required_features if f not in features]
            
            if missing_features:
                return jsonify({
                    "success": False,
                    "error": f"Missing required features: {missing_features}",
                    "expected_features": required_features
                }), 400
        else:
            return jsonify({
                "success": False,
                "error": "Invalid request format. Send either 'data' (raw time series) or 'features' (pre-calculated)",
                "example_data_format": {"data": [1000000, 1100000, 1050000, 1200000], "weeks": 1},
                "example_features_format": {"features": {"lag_1": 1200000, "lag_2": 1050000, "lag_3": 1100000, "ma_4": 1087500, "bulan": 12, "minggu_dalam_bulan": 4}}
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
            "calculated_features": features
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid value: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
