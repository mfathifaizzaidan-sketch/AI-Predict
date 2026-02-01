from http.server import BaseHTTPRequestHandler
import json
import pickle
import numpy as np
import os

# Load model at module level for better performance
model_path = os.path.join(os.path.dirname(__file__), '..', 'model_prediksi_saldo_mingguan.pkl')
model = None

def load_model():
    global model
    if model is None:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    return model

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight request"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET request - return API info"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
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
        }
        self.wfile.write(json.dumps(response, indent=2).encode())

    def do_POST(self):
        """Handle POST request - make prediction"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Extract features
            features = data.get('features', {})
            
            # Validate required features
            required_features = ['week_number', 'total_waste_kg', 'num_transactions', 'num_active_nasabah']
            missing_features = [f for f in required_features if f not in features]
            
            if missing_features:
                self.send_error_response(400, f"Missing required features: {missing_features}")
                return
            
            # Prepare features array for prediction
            feature_values = [
                float(features['week_number']),
                float(features['total_waste_kg']),
                float(features['num_transactions']),
                float(features['num_active_nasabah'])
            ]
            
            # Load model and make prediction
            loaded_model = load_model()
            X = np.array([feature_values])
            prediction = loaded_model.predict(X)[0]
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "success": True,
                "prediction": float(prediction),
                "message": "Prediksi saldo mingguan berhasil",
                "input_features": features
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON format")
        except Exception as e:
            self.send_error_response(500, f"Prediction error: {str(e)}")
    
    def send_error_response(self, status_code, message):
        """Helper to send error response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "success": False,
            "error": message
        }
        self.wfile.write(json.dumps(response, indent=2).encode())
