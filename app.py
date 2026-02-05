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

def extract_values(data):
    """
    Extract numeric values from various data formats:
    - Plain numbers: [1000, 2000, 3000]
    - Objects with 'saldo': [{"tanggal": "...", "saldo": 1000}, ...]
    - Objects with 'value': [{"date": "...", "value": 1000}, ...]
    """
    values = []
    for item in data:
        if isinstance(item, (int, float)):
            values.append(float(item))
        elif isinstance(item, dict):
            # Try different possible field names
            if 'saldo' in item:
                values.append(float(item['saldo']))
            elif 'Saldo' in item:
                values.append(float(item['Saldo']))
            elif 'value' in item:
                values.append(float(item['value']))
            elif 'amount' in item:
                values.append(float(item['amount']))
            elif 'balance' in item:
                values.append(float(item['balance']))
            else:
                # Try to get the first numeric value in the dict
                for v in item.values():
                    if isinstance(v, (int, float)):
                        values.append(float(v))
                        break
        elif isinstance(item, str):
            # Try to parse string as number
            try:
                values.append(float(item.replace(',', '').replace('.', '')))
            except ValueError:
                pass
    return values

def calculate_features(data, weeks=1):
    """
    Calculate lag features from time series data
    data: list of weekly balance values (most recent last)
    Returns: features dict with lag_1, lag_2, lag_3, ma_4, bulan, minggu_dalam_bulan
    """
    # Extract numeric values from data
    values = extract_values(data)
    
    if len(values) < 4:
        raise ValueError(f"Need at least 4 weeks of data, got {len(values)}")
    
    # Get the last 4 values for lag features
    last_values = values[-4:]
    
    # lag_1 = most recent value
    # lag_2 = second most recent
    # lag_3 = third most recent
    lag_1 = last_values[-1]
    lag_2 = last_values[-2]
    lag_3 = last_values[-3]
    
    # ma_4 = 4-week moving average
    ma_4 = sum(last_values) / 4
    
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
                "format": "raw_data_objects",
                "description": "Send weekly data as objects",
                "example": {
                    "data": [
                        {"tanggal": "2025-11-21", "saldo": 763008},
                        {"tanggal": "2025-11-30", "saldo": 2929676}
                    ],
                    "weeks": 1
                }
            },
            {
                "format": "raw_data_numbers",
                "description": "Send weekly data as numbers",
                "example": {
                    "data": [763008, 2929676, 1669650, 1500000],
                    "weeks": 1
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
            
            if not raw_data:
                return jsonify({
                    "success": False,
                    "error": "Empty 'data' array provided"
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
                "error": "Invalid request format. Send 'data' array with weekly balance data.",
                "example": {"data": [{"tanggal": "2025-11-21", "saldo": 763008}], "weeks": 1}
            }), 400
        
        # Prepare features array for prediction
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
            "error": f"Data error: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
