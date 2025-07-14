from typing import Tuple, Annotated
import pandas as pd
from zenml.steps import step
from src.data_splitter import (
    ForecastingDataSplitter,
    ClassificationDataSplitter,
    RecommendationDataSplitter
)


@step
def forecasting_data_split_step(
    data: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "ts_train"],
    Annotated[pd.DataFrame, "ts_val"]
]:
    splitter = ForecastingDataSplitter()
    ts_train, ts_val = splitter.split(data)
    return ts_train, ts_val


@step
def classification_data_split_step(
    data: pd.DataFrame,
    y_column: str
) -> Tuple[
    Annotated[pd.DataFrame, "X_cls_train"],
    Annotated[pd.DataFrame, "X_cls_test"],
    Annotated[pd.Series, "y_cls_train"],
    Annotated[pd.Series, "y_cls_test"]
]:
    splitter = ClassificationDataSplitter()
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = splitter.split(data, y_column)
    return X_cls_train, X_cls_test, y_cls_train, y_cls_test


@step
def recommendation_data_split_step(
    data: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "rec_train"],
    Annotated[pd.DataFrame, "rec_test"]
]:
    splitter = RecommendationDataSplitter()
    rec_train, rec_test = splitter.split(data)
    return rec_train, rec_test
