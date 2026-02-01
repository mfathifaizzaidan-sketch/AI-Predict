from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

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

@app.route('/api/predict', methods=['GET'])
def predict_info():
    """GET - return API info"""
    return jsonify({
        "success": True,
        "message": "AI Prediction API untuk Bank Sampah OSKU",
        "version": "1.0.0",
        "endpoint": "/api/predict",
        "method": "POST",
        "expected_features": [
            "week_number",
            "total_waste_kg", 
            "num_transactions",
            "num_active_nasabah"
        ],
        "example_request": {
            "features": {
                "week_number": 5,
                "total_waste_kg": 150.5,
                "num_transactions": 25,
                "num_active_nasabah": 10
            }
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """POST - make prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Extract features
        features = data.get('features', {})
        
        # Validate required features
        required_features = ['week_number', 'total_waste_kg', 'num_transactions', 'num_active_nasabah']
        missing_features = [f for f in required_features if f not in features]
        
        if missing_features:
            return jsonify({
                "success": False,
                "error": f"Missing required features: {missing_features}"
            }), 400
        
        # Prepare features array for prediction
        feature_values = [
            float(features['week_number']),
            float(features['total_waste_kg']),
            float(features['num_transactions']),
            float(features['num_active_nasabah'])
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
