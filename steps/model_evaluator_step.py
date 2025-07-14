from typing import Annotated
import pandas as pd
from zenml import step
from src.model_evaluator import (
    TimeSeriesEvaluation,
    ClassificationEvaluation,
    RecommendationEvaluation
)
import mlflow

@step(enable_cache=False)
def evaluate_time_series_model(trained_model, df_val: pd.DataFrame) -> Annotated[dict, "Forecast Evaluation Metrics"]:
    metrics  = TimeSeriesEvaluation().evaluate_model(trained_model, df_val)
    if not mlflow.active_run():
        mlflow.start_run()
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
    mlflow.end_run()
    return metrics

@step(enable_cache=False)
def evaluate_classification_model(trained_model, X_test: pd.DataFrame, y_test: pd.Series) -> Annotated[dict, "Classification Report"]:
    metrics = ClassificationEvaluation().evaluate_model(trained_model, X_test, y_test)
    if not mlflow.active_run():
        mlflow.start_run()
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
    mlflow.end_run()
    return metrics

@step(enable_cache=False)
def evaluate_recommendation_model(trained_model: dict, rec_test) -> Annotated[dict, "Recommendation Evaluation"]:
    metrics = RecommendationEvaluation().evaluate_model(trained_model, rec_test)
    if not mlflow.active_run():
        mlflow.start_run()
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
    mlflow.end_run()
    return metrics