from zenml import pipeline
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from steps.dynamic_loader import dynamic_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import forecast_predictor, classification_predictor, recommendation_predictor
import json
import pandas as pd

@pipeline
def continuous_deployment_pipeline():
    
    models = ml_pipeline()
    
    mlflow_model_deployer_step(
        model = models['forecast_model'],
        model_name = "forecasting_model",
        deploy_decision = True,
    )

    mlflow_model_deployer_step(
        model = models['classification_model'],
        model_name = "classification_model",
        deploy_decision = True
    )

    mlflow_model_deployer_step(
        model = models['recommendation_model'],
        model_name = "recommendation_model",
        deploy_decision = True
    )

@pipeline(enable_cache=False)
def inference_pipeline(model_type: str):
    input_data = dynamic_loader(model_type=model_type)

    model_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
        model_name=f"{model_type}_model",
    )

    if model_type == "forecasting":
        forecast_predictor(service=model_service, input_data=input_data)

    elif model_type == "classification":
        classification_predictor(service=model_service, input_data=input_data)

    elif model_type == "recommendation":
        recommendation_predictor(service=model_service, user_vector=input_data)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
