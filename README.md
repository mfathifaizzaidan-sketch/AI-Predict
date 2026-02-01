# AI Prediction API - Bank Sampah OSKU

API endpoint untuk prediksi saldo mingguan menggunakan model Machine Learning (Random Forest).

## 🚀 Deployment

API ini di-deploy menggunakan **Railway** (atau platform serupa seperti Render, Heroku).

### Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Info API |
| GET | `/health` | Health check |
| GET | `/api/predict` | Info endpoint prediksi |
| POST | `/api/predict` | Melakukan prediksi |

## 📋 Cara Penggunaan

### GET Request - Info API

```bash
curl https://[your-railway-domain]/api/predict
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
curl -X POST https://[your-railway-domain]/api/predict \
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
├── app.py                      # Flask application
├── api/
│   └── predict.py              # Vercel endpoint (backup)
├── model_prediksi_saldo_mingguan.pkl  # Model ML
├── requirements.txt            # Dependencies
├── Procfile                    # Railway/Heroku config
├── pyproject.toml             # Python project config
├── vercel.json                # Vercel config (backup)
├── .gitignore                 # Ignore files
└── README.md                  # Dokumentasi
```

## 🛠️ Deploy ke Railway

1. Buka [railway.app](https://railway.app)
2. Login dengan GitHub
3. Klik "New Project" → "Deploy from GitHub repo"
4. Pilih repository `AI-Predict`
5. Railway akan auto-detect dan deploy
6. Klik "Generate Domain" untuk mendapatkan URL public

## 🧪 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py

# Test endpoint
curl http://localhost:5000/api/predict
```

## 📝 License

MIT License - Bank Sampah OSKU

