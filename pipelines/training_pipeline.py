import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steps.data_ingestion_step import data_ingestion_step
from steps.EDA_step import eda_step
from steps.data_cleaning_step import data_cleaning_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import (
    forecasting_data_split_step,
    classification_data_split_step,
    recommendation_data_split_step,
)
from steps.model_building_step import (
    train_timeseries_model_step,
    train_classification_model_step,
    train_recommendation_model_step
)
from steps.model_evaluator_step import (
    evaluate_time_series_model,
    evaluate_classification_model,
    evaluate_recommendation_model
)
from zenml import Model, pipeline

@pipeline(
        name="ML_pipeline",
        model=Model(
            name="retail_store_model"
        ),
        enable_cache=False
)
def ml_pipeline():
    raw_data = data_ingestion_step('sql')
    report_path = eda_step(raw_data)
    cleaned_customers, cleaned_products, cleaned_sales = data_cleaning_step(raw_data)

    forecasting_df, classification_df, recommendation_df = feature_engineering_step(
        cleaned_customers,
        cleaned_products,
        cleaned_sales
    )

    ts_train, ts_val = forecasting_data_split_step(forecasting_df)

    X_cls_train, X_cls_test, y_cls_train, y_cls_test = classification_data_split_step(
        classification_df, y_column="label"
    )

    rec_train, rec_test = recommendation_data_split_step(recommendation_df)

    ts_model = train_timeseries_model_step(ts_train, ts_val)
    cls_model = train_classification_model_step(X_cls_train, y_cls_train)
    rec_model = train_recommendation_model_step(rec_train=rec_train, cleaned_data=cleaned_products)

    time_series_evaluation = evaluate_time_series_model(ts_model, ts_val)
    classification_evaluation = evaluate_classification_model(cls_model, X_cls_test, y_cls_test)
    recommender_evaluation = evaluate_recommendation_model(rec_model, rec_test)

    return {
        "forecast_model": ts_model,
        "classification_model": cls_model,
        "recommendation_model": rec_model,
    }

# if __name__ == "__main__":
#     ml_pipeline()
