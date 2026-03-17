# 🧠 Banknote Authentication API

This project is a Machine Learning API that predicts whether a banknote is genuine or fake using a Random Forest model.

## 🚀 Features
- FastAPI backend
- Dockerized application
- ML model using scikit-learn
- REST API for predictions

## 📦 Tech Stack
- Python
- FastAPI
- Scikit-learn
- Docker

## 📊 Input Features
- Variance
- Skewness
- Curtosis
- Entropy

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload


## Run with Docker
```bash
docker build -t banknote-api .
docker run -p 8000:8000 banknote-api





📡 API Endpoint

POST /predict

{
  "variance": 2.3,
  "skewness": 6.7,
  "curtosis": -1.2,
  "entropy": 0.5
}



📌 Output
{
  "prediction": 0,
  "result": "Genuine Note",
  "confidence": 0.98
}
