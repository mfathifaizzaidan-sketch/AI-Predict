"""
Script to regenerate the Random Forest model for weekly sales prediction.
This creates a fresh pickle file compatible with scikit-learn 1.6.1 and Python 3.11.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Create a simple Random Forest model for weekly balance prediction
# Features: week_number, total_waste_kg, num_transactions, num_active_nasabah

# Generate synthetic training data based on realistic patterns
np.random.seed(42)

n_samples = 200

# Generate features
week_numbers = np.random.randint(1, 53, n_samples)  # Week 1-52
total_waste_kg = np.random.uniform(50, 500, n_samples)  # 50-500 kg per week
num_transactions = np.random.randint(10, 100, n_samples)  # 10-100 transactions
num_active_nasabah = np.random.randint(5, 50, n_samples)  # 5-50 active users

# Create feature matrix
X = np.column_stack([
    week_numbers,
    total_waste_kg,
    num_transactions,
    num_active_nasabah
])

# Generate target (weekly balance) with some realistic formula + noise
# Balance = base_rate * waste_kg + transaction_factor * transactions + noise
base_rate = 500  # Rp per kg
transaction_bonus = 1000  # Rp per transaction
y = (base_rate * total_waste_kg + 
     transaction_bonus * num_transactions +
     np.random.normal(0, 10000, n_samples))  # Add some noise

# Train the model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")
model.fit(X, y)

# Test prediction
test_features = np.array([[5, 150.5, 25, 10]])  # Example input
test_prediction = model.predict(test_features)[0]
print(f"Test prediction for week 5, 150.5 kg, 25 transactions, 10 nasabah: Rp {test_prediction:,.0f}")

# Save the model
model_path = 'model_prediksi_saldo_mingguan.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nModel saved to {model_path}")
print(f"File size: {__import__('os').path.getsize(model_path):,} bytes")

# Verify the model can be loaded
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)
    verify_prediction = loaded_model.predict(test_features)[0]
    print(f"Verification prediction: Rp {verify_prediction:,.0f}")
    print("Model verification successful!" if abs(verify_prediction - test_prediction) < 0.01 else "Verification FAILED!")
