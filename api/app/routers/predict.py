from fastapi import APIRouter
from app.predict import predict_fraud

router = APIRouter()

# POST method to run prediction using the run_prediction function that loads our model
@router.post("/", tags=["Prediction"])
def run_prediction(features: dict):
    result = predict_fraud(features)
    return result