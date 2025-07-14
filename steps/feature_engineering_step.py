from typing import Annotated
import pandas as pd
from zenml import step
from src.feature_engineering import (
    SalesForecastingFeatureEngineer,
    CategoryClassificationEngineer,
    RecommendationFeatureEngineer
)

@step
def feature_engineering_step(
    cleaned_customers: pd.DataFrame,
    cleaned_products: pd.DataFrame,
    cleaned_sales: pd.DataFrame
) -> tuple[
    Annotated[pd.DataFrame, "forecasting_df"],
    Annotated[pd.DataFrame, "classification_df"],
    Annotated[pd.DataFrame, "recommendation_df"]
]:
    data = {
        "customers": cleaned_customers,
        "products": cleaned_products,
        "sales": cleaned_sales
    }

    forecasting = SalesForecastingFeatureEngineer().engineer_features(data)
    classification = CategoryClassificationEngineer().engineer_features(data)
    recommendation = RecommendationFeatureEngineer().engineer_features(data)

    return forecasting, classification, recommendation

