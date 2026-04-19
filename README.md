
# 🌸 MLOps Mini Project — Iris Classifier API

**Author:** HIMANSHU CHAUHAN
**Enrollment No:** 01618012723
**Assignment:** MLOps Workshop — Assignment 2

---

## 📌 Project Overview
End-to-end MLOps pipeline for Iris flower classification with:
- REST API built with FastAPI
- CI/CD automation via GitHub Actions
- Containerization using Docker
- Logging and system monitoring

## 📁 Project Structure
```
mlops-mini-project/
├── data/                    # raw data (if any)
├── models/                  # saved trained model
│   ├── iris_model.joblib
│   └── target_names.npy
├── src/
│   ├── train.py             # model training script
│   └── test_api.py          # automated API tests
├── .github/
│   └── workflows/
│       └── ci_cd.yml        # GitHub Actions CI/CD pipeline
├── app.py                   # FastAPI application
├── Dockerfile               # Docker container config
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
```

### 3. Start the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🐳 How to Run with Docker

```bash
# Build image
docker build -t iris-mlops-api .

# Run container
docker run -p 8000:8000 iris-mlops-api

# Test
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check (used in CI/CD) |
| GET | `/metrics` | CPU/RAM monitoring |
| POST | `/predict` | Predict iris species |
| GET | `/docs` | Swagger UI |

## 📊 Example Request

```json
POST /predict
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

```json
Response:
{
  "predicted_class": "setosa",
  "class_index": 0,
  "confidence": "99.8%",
  "all_probabilities": {
    "setosa": 99.8,
    "versicolor": 0.1,
    "virginica": 0.1
  }
}
```

## ⚙️ CI/CD Pipeline
GitHub Actions automatically:
1. Installs all dependencies
2. Runs the training script
3. Starts the API server
4. Runs all API tests
5. Fails the build if any test fails

## ⭐ Bonus Features
- Request logging with timestamps
- `/metrics` endpoint for CPU and RAM monitoring
- Health check endpoint for CI/CD
- Structured logging in train.py
