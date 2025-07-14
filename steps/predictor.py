from zenml import step
import pandas as pd
import numpy as np
from typing import List
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService


@step
def forecast_predictor(service: MLFlowDeploymentService, input_data: pd.DataFrame) -> List[float]:
    """Runs prediction on forecasting model."""
    service.start(timeout=10)
    predictions = service.predict(input_data.to_dict(orient="records"))
    return predictions


@step
def classification_predictor(service: MLFlowDeploymentService, input_data: pd.DataFrame) -> List[str]:
    """Runs prediction on classification model."""
    service.start(timeout=10)
    predictions = service.predict(input_data.to_dict(orient="records"))
    return predictions


@step
def recommendation_predictor(service: MLFlowDeploymentService, user_vector: List[float]) -> List[float]:
    """Runs prediction on recommendation model."""
    service.start(timeout=10)
    # Ensure it's 2D for single row
    user_array = np.array([user_vector]) if isinstance(user_vector[0], float) else np.array(user_vector)
    predictions = service.predict(user_array)
    return predictions
