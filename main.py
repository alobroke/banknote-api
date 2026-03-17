from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel, Field

app = FastAPI(title="Bank Note Authentication API")

# Load trained model
with open("classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema (with validation)
from pydantic import BaseModel, ConfigDict

class BankNoteInput(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "variance": 2.3,
                "skewness": 6.7,
                "curtosis": -1.2,
                "entropy": 0.5
            }
        }
    )

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: BankNoteInput):
    try:
        # Convert input to numpy array
        features = np.array([
            data.variance,
            data.skewness,
            data.curtosis,
            data.entropy
        ]).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()

        return {
            "prediction": int(prediction),
            "result": "Fake Note" if prediction == 1 else "Genuine Note",
            "confidence": round(float(probability), 3)
        }

    except Exception as e:
        return {"error": str(e)}