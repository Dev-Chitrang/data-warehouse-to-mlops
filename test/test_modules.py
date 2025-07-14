import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from dotenv import load_dotenv

from src.ingest_data import SQLServerIngestor
from src.EDA import YDataProfilingEDA
from src.data_cleaning import CustomersFormator, ProductFormator, SalesFormator
from src.feature_engineering import (
    SalesForecastingFeatureEngineer,
    CategoryClassificationEngineer,
    RecommendationFeatureEngineer
)
from src.data_splitter import TimeSeriesSplitter, StandardSplitter
from src.model_building import TrainTimeSeriesModel, TrainClassificationModel, TrainRecommendationModel
from src.model_evaluator import TimeSeriesEvaluation, ClassificationEvaluation, RecommendationEvaluation

load_dotenv(".env")
DRIVER = os.getenv("DRIVER")
SERVER = os.getenv("SERVER")
DATABASE = os.getenv("DATABASE")
TRUST_CONNECTION = os.getenv("TRUST_CONNECTION", "true")
TABLE_NAMES = os.getenv("TABLE_NAMES", "").split(",")
trusted_conn_value = "yes" if TRUST_CONNECTION.lower() == "true" else "no"
CONNECTION_STRING = (
    f"DRIVER={DRIVER};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection={trusted_conn_value}"
)

sql_ingestor = SQLServerIngestor(CONNECTION_STRING, TABLE_NAMES)
data = sql_ingestor.ingest_data()
for table, df in data.items():
    print(f"Table: {table}, Rows: {len(df)}")
print("Data loaded successfully!")

report_paths = YDataProfilingEDA().generate_report(data=data)
print("EDA report generated successfully!")
print("Report Path: {report_paths}")

cleaned_data = {
    "customers": CustomersFormator().transform(data['customers']),
    "products": ProductFormator().transform(data['products']),
    "sales": SalesFormator().transform(data['sales'])
}
print("Data cleaned successfully!")

featured_data = {
    "forecasting": SalesForecastingFeatureEngineer().engineer_features(cleaned_data),
    "classification": CategoryClassificationEngineer().engineer_features(cleaned_data),
    "recommendation": RecommendationFeatureEngineer().engineer_features(cleaned_data)
}
print("Features engineered successfully!")

x_cls = featured_data['classification'].drop(columns=['label'])
y_cls = featured_data['classification']['label']
X_cls_train, X_cls_test, y_cls_train, y_cls_test = StandardSplitter().split_data(x_cls, y_cls)

rec_train, rec_test = StandardSplitter().split_data(featured_data['recommendation'])

ts_train, ts_val = TimeSeriesSplitter().split_data(featured_data['forecasting'])
print("Data Splitting Completed!")

trained_models = {
    "forecasting": TrainTimeSeriesModel().build_and_train_model(ts_train, ts_val),
    "classification": TrainClassificationModel().build_and_train_model(X_cls_train, y_cls_train),
    "recommendation": TrainRecommendationModel().build_and_train_model(rec_train, cleaned_data["products"])
}
print("Model Training Completed!")

evaluations = {
    "forecasting": TimeSeriesEvaluation().evaluate_model(trained_models["forecasting"], ts_val, None),
    "classification": ClassificationEvaluation().evaluate_model(
        trained_models["classification"], X_cls_test, y_cls_test
    ),
    "recommendation": RecommendationEvaluation().evaluate_model(
        trained_models["recommendation"], rec_test
    )
}

for model_name, evaluation in evaluations.items():
    print(f"Model: {model_name}")
    for metric, value in evaluation.items():
        print(f"{metric}: {value}")

print("Model Evaluation Completed!")