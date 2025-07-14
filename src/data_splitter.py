from abc import ABC, abstractmethod
from typing import Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from neuralprophet import NeuralProphet


class BaseDataSplitter(ABC):
    @abstractmethod
    def split(self, *args, **kwargs):
        pass


class ForecastingDataSplitter(BaseDataSplitter):
    def __init__(self, valid_p: float = 0.2, freq: str = "D"):
        self.valid_p = valid_p
        self.freq = freq

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "ds" not in data or "y" not in data:
            raise ValueError("Time series data must contain 'ds' and 'y' columns.")
        data = data.copy()
        data["ds"] = pd.to_datetime(data["ds"])
        data = data.sort_values("ds")

        model = NeuralProphet()
        df_train, df_val = model.split_df(data, freq=self.freq, valid_p=self.valid_p)
        return df_train, df_val

class ClassificationDataSplitter(BaseDataSplitter):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, data: pd.DataFrame, y_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = data.drop(columns=[y_column])
        y = data[y_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)


class RecommendationDataSplitter(BaseDataSplitter):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, user_item_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Randomly splits users into train and test groups based on their interaction matrix."""
        user_ids = user_item_matrix.index.to_numpy()
        train_users, test_users = train_test_split(user_ids, test_size=self.test_size, random_state=self.random_state)

        train_df = user_item_matrix.loc[train_users]
        test_df = user_item_matrix.loc[test_users]

        return train_df, test_df