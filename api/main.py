import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.build_features import create_features


# ------------------------------------------------
# Configuration
# ------------------------------------------------

MODEL_PATH = "artifacts/model/model.joblib"


# ------------------------------------------------
# FastAPI application
# ------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn.",
    version="1.0.0"
)


# ------------------------------------------------
# Load model
# ------------------------------------------------

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


# ------------------------------------------------
# Input schema
# ------------------------------------------------

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ------------------------------------------------
# Root endpoint
# ------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


# ------------------------------------------------
# Health endpoint
# ------------------------------------------------

@app.get("/health")
def health():
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ------------------------------------------------
# Model information endpoint
# ------------------------------------------------

@app.get("/model-info")
def model_info():
    return {
        "model": "Decision Tree",
        "problem": "Customer Churn Prediction",
        "selection_metric": "F1 Score",
        "f1_score": 0.5989
    }


# ------------------------------------------------
# Prediction endpoint
# ------------------------------------------------

@app.post("/predict")
def predict_customer_churn(customer: CustomerData):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model file not found."
        )

    try:
        input_data = pd.DataFrame([customer.model_dump()])

        # Apply the same feature engineering used during training
        input_data = create_features(input_data)

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        churn_prediction = "Yes" if prediction == 1 else "No"

        return {
            "churn_prediction": churn_prediction,
            "churn_probability": round(float(probability), 4)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )