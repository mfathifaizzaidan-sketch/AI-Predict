"""
Script to regenerate the Random Forest model for weekly sales prediction.
This creates a fresh pickle file compatible with scikit-learn 1.6.1 and numpy < 2.0.
Features: lag_1, lag_2, lag_3, ma_4, bulan, minggu_dalam_bulan
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest model for weekly balance prediction
# Features: lag_1, lag_2, lag_3, ma_4, bulan, minggu_dalam_bulan

np.random.seed(42)

n_samples = 500

# Generate synthetic training data based on realistic patterns
# Simulate weekly balance data with lag features

# Base balance around 1-3 million Rupiah
base_balance = np.random.uniform(500000, 3000000, n_samples)

# lag_1: previous week balance (highly correlated with target)
lag_1 = base_balance * np.random.uniform(0.9, 1.1, n_samples)

# lag_2: 2 weeks ago balance  
lag_2 = lag_1 * np.random.uniform(0.85, 1.15, n_samples)

# lag_3: 3 weeks ago balance
lag_3 = lag_2 * np.random.uniform(0.8, 1.2, n_samples)

# ma_4: 4-week moving average
ma_4 = (lag_1 + lag_2 + lag_3 + base_balance) / 4

# bulan: month (1-12)
bulan = np.random.randint(1, 13, n_samples)

# minggu_dalam_bulan: week within month (1-5)
minggu_dalam_bulan = np.random.randint(1, 6, n_samples)

# Create feature matrix
X = np.column_stack([
    lag_1,
    lag_2,
    lag_3,
    ma_4,
    bulan,
    minggu_dalam_bulan
])

# Generate target (next week's balance) - influenced by lags and seasonal patterns
# Balance tends to follow previous weeks with some growth and seasonality
seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * bulan / 12)  # Seasonal variation
trend = 1 + 0.02 * minggu_dalam_bulan  # Slight weekly trend

y = (0.5 * lag_1 + 0.3 * lag_2 + 0.2 * ma_4) * seasonal_factor * trend
y = y + np.random.normal(0, 50000, n_samples)  # Add noise

# Ensure no negative values
y = np.maximum(y, 100000)

# Train the model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")
print(f"Features: lag_1, lag_2, lag_3, ma_4, bulan, minggu_dalam_bulan")
print(f"Training samples: {n_samples}")
model.fit(X, y)

# Test prediction
test_features = np.array([[1500000, 1400000, 1300000, 1400000, 12, 4]])  # Example input
test_prediction = model.predict(test_features)[0]
print(f"\nTest prediction:")
print(f"  Input: lag_1=1.5M, lag_2=1.4M, lag_3=1.3M, ma_4=1.4M, bulan=12, minggu=4")
print(f"  Predicted balance: Rp {test_prediction:,.0f}")

# Feature importance
print(f"\nFeature Importance:")
feature_names = ['lag_1', 'lag_2', 'lag_3', 'ma_4', 'bulan', 'minggu_dalam_bulan']
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"  {name}: {importance:.4f}")

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
