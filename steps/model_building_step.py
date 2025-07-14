import logging
from typing import Annotated
import joblib
import os
from datetime import datetime
import mlflow
import pandas as pd
from zenml import step, ArtifactConfig
from zenml.client import Client
from zenml import Model
from src.model_building import (
    TrainTimeSeriesModel,
    TrainClassificationModel,
    TrainRecommendationModel
)
from neuralprophet import NeuralProphet
from sklearn.ensemble import RandomForestClassifier
from zenml.materializers import BuiltInContainerMaterializer
from wrapper.neualprophet_wrapper import NeuralProphetModel

experiment_tracker = Client().active_stack.experiment_tracker

# -------------------------------
# Timeseries Model Step
# -------------------------------
forecasting_model = Model(
    name="Sales Forecasting Model",
    description="NeuralProphet model for sales forecasting.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=forecasting_model)
def train_timeseries_model_step(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> Annotated[NeuralProphet, ArtifactConfig(name="forecasting_model", is_model_artifact=True)]:
    """Train NeuralProphet forecasting model."""

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.set_tag("model_type", "NeuralProphet")
        logging.info("Training NeuralProphet forecasting model.")
        model = TrainTimeSeriesModel().build_and_train_model(df_train, df_val)
        code_path = [os.path.abspath("wrapper")]
        mlflow.pyfunc.log_model(
            artifact_path="forecasting_model",
            python_model=NeuralProphetModel(model),
            code_path=code_path
        )
        return model

    except Exception as e:
        logging.error(f"Forecasting training failed: {e}")
        raise e

    finally:
        mlflow.end_run()


# -------------------------------
# Classification Model Step
# -------------------------------
classification_model = Model(
    name="Category Classification Model",
    description="RandomForestClassifier for customer category prediction.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=classification_model)
def train_classification_model_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[RandomForestClassifier, ArtifactConfig(name="classification_model", is_model_artifact=True)]:
    """Train RandomForest classification model."""

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        logging.info("Training RandomForest classifier.")
        model = TrainClassificationModel().build_and_train_model(X_train, y_train)
        mlflow.sklearn.log_model(model, artifact_path="classification_model")
        return model

    except Exception as e:
        logging.error(f"Classification training failed: {e}")
        raise e

    finally:
        mlflow.end_run()


# -------------------------------
# Recommendation Model Step
# -------------------------------
recommendation_model = Model(
    name="Recommendation Model",
    description="Hybrid collaborative + content-based recommender.",
)

@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
    model=recommendation_model,
    output_materializers={"recommendation_model": BuiltInContainerMaterializer},
)
def train_recommendation_model_step(
    rec_train: pd.DataFrame,
    cleaned_data: pd.DataFrame
) -> Annotated[dict, ArtifactConfig(name="recommendation_model", is_model_artifact=True)]:
    """Train hybrid recommendation model."""
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.set_tag("model_type", "Hybrid Recommendation")
        logging.info("Training hybrid recommendation model.")
        model = TrainRecommendationModel().build_and_train_model(
            rec_train, cleaned_data
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"recommendation_model_{timestamp}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        return model

    except Exception as e:
        logging.error(f"Recommendation training failed: {e}")
        raise e

    finally:
        mlflow.end_run()
        if os.path.exists(model_path):
            os.remove(model_path)
