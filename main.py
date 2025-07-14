from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI()

# Paths to MLflow models
CLASSIFICATION_MODEL_PATH = "mlruns/870593124420468070/53a28cbb3d5b4fb7930183f1b5fe2eab/artifacts/classification_model"
FORECASTING_MODEL_PATH    = "mlruns/870593124420468070/53a28cbb3d5b4fb7930183f1b5fe2eab/artifacts/forecasting_model"
RECOMMENDER_MODEL_PATH    = "mlruns/870593124420468070/53a28cbb3d5b4fb7930183f1b5fe2eab/artifacts/recommendation_model_20250712_160912.pkl"

# Load models
classification_model = mlflow.pyfunc.load_model(CLASSIFICATION_MODEL_PATH)
forecasting_model = mlflow.pyfunc.load_model(FORECASTING_MODEL_PATH)
import joblib
recommender_model = joblib.load(RECOMMENDER_MODEL_PATH)

@app.post("/predict/classification")
def predict_classification(data: list[list[float]]):
    df = pd.DataFrame(data)
    preds = classification_model.predict(df)
    return {"predictions": preds.tolist()}

@app.post("/predict/forecasting")
def predict_forecasting(data: list[dict]):
    df = pd.DataFrame(data)
    preds = forecasting_model.predict(df)
    return preds.to_dict(orient="records")

@app.post("/predict/recommendation")
def predict_recommendation(data: list[float]):
    import numpy as np
    arr = np.array(data).reshape(1, -1)
    preds = recommender_model.predict(arr)
    return {"recommendations": preds.tolist()}
