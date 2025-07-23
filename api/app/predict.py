import joblib 
import pandas as pd 
import os 
import yaml 
from app.kafka_consumer import load_config

# load the config file and extract the model path
config_path = '/app/config.yaml'
config = load_config(config_path)
model_path = config['model']['path']

# load the trained model 
model = joblib.load(model_path)

# main prediction function 
def predict_fraud(features: dict):
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0][1]
    return {
        "is_fraud": bool(proba > 0.5),
        "probability": float(round(proba, 3))
    }