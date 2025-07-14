import pandas as pd
from abc import ABC, abstractmethod
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from neuralprophet import NeuralProphet
from sklearn.preprocessing import OneHotEncoder

class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        pass

class TrainTimeSeriesModel(ModelBuildingStrategy):
    def build_and_train_model(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Any:
        if not isinstance(df_train, pd.DataFrame) or not isinstance(df_val, pd.DataFrame):
            raise TypeError("Expected pandas DataFrames for df_train and df_val.")
        model = NeuralProphet(epochs=100)
        model.fit(df_train)
        return model

class TrainClassificationModel(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
            raise TypeError("Expected DataFrame for X_train and Series for y_train")

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model


class TrainRecommendationModel(ModelBuildingStrategy):
    def build_and_train_model(self, user_item_matrix: pd.DataFrame, products: pd.DataFrame) -> dict:
        svd = TruncatedSVD(n_components=20, random_state=42)
        user_embeddings = svd.fit_transform(user_item_matrix)
        item_embeddings = svd.components_.T

        product_features = products.set_index("product_key")[["category"]].fillna("Unknown")
        encoder = OneHotEncoder()
        product_feature_matrix = encoder.fit_transform(product_features).toarray()
        product_keys = product_features.index.tolist()
        content_similarity = cosine_similarity(product_feature_matrix)

        model_dict = {
            "svd_model": svd,
            "user_embeddings": user_embeddings,
            "item_embeddings": item_embeddings,
            "user_item_matrix": user_item_matrix,
            "product_features": product_features,
            "product_feature_matrix": product_feature_matrix,
            "product_keys": product_keys,
            "cosine_similarity": content_similarity,
            "encoder": encoder
        }

        return model_dict
