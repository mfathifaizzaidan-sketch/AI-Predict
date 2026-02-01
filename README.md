# AI Prediction API - Bank Sampah OSKU

API endpoint untuk prediksi saldo mingguan menggunakan model Machine Learning (Random Forest).

## 🚀 Deployment

API ini di-deploy menggunakan Vercel Python Serverless Functions.

### Endpoint

- **URL**: `https://[your-vercel-domain]/api/predict`
- **Method**: `POST` (untuk prediksi) / `GET` (untuk info API)

## 📋 Cara Penggunaan

### GET Request - Info API

```bash
curl https://[your-vercel-domain]/api/predict
```

**Response:**
```json
{
  "success": true,
  "message": "AI Prediction API untuk Bank Sampah OSKU",
  "version": "1.0.0",
  "endpoint": "/api/predict",
  "method": "POST",
  "expected_features": [
    "week_number",
    "total_waste_kg",
    "num_transactions", 
    "num_active_nasabah"
  ]
}
```

### POST Request - Prediksi

```bash
curl -X POST https://[your-vercel-domain]/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "week_number": 5,
      "total_waste_kg": 150.5,
      "num_transactions": 25,
      "num_active_nasabah": 10
    }
  }'
```

**Response Success:**
```json
{
  "success": true,
  "prediction": 1250000.50,
  "message": "Prediksi saldo mingguan berhasil",
  "input_features": {
    "week_number": 5,
    "total_waste_kg": 150.5,
    "num_transactions": 25,
    "num_active_nasabah": 10
  }
}
```

**Response Error:**
```json
{
  "success": false,
  "error": "Missing required features: ['week_number']"
}
```

## 📊 Input Features

| Feature | Type | Deskripsi |
|---------|------|-----------|
| `week_number` | number | Nomor minggu dalam tahun (1-52) |
| `total_waste_kg` | number | Total sampah yang dikumpulkan (kg) |
| `num_transactions` | number | Jumlah transaksi pada minggu tersebut |
| `num_active_nasabah` | number | Jumlah nasabah aktif |

## ⚙️ Struktur Project

```
AI-Predict/
├── api/
│   └── predict.py              # Endpoint /api/predict
├── model_prediksi_saldo_mingguan.pkl  # Model ML
├── requirements.txt            # Dependencies
├── vercel.json                # Konfigurasi Vercel
├── .gitignore                 # Ignore files
└── README.md                  # Dokumentasi
```

## 🛠️ Development

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Test Model
```python
import pickle
import numpy as np

# Load model
with open('model_prediksi_saldo_mingguan.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
X = np.array([[5, 150.5, 25, 10]])  # [week, waste_kg, transactions, nasabah]
prediction = model.predict(X)
print(f"Prediksi: Rp {prediction[0]:,.2f}")
```

## 📝 License

MIT License - Bank Sampah OSKU
