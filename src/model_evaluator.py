from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report

class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model, X_test, y_test)->dict:
        pass

class TimeSeriesEvaluation:
    @staticmethod
    def evaluate_model(model, df_val, true_vals=None):
        forecast = model.predict(df_val)

        y_true = forecast["y"]
        y_pred = forecast["yhat1"]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        return {
            "MAE": round(mae, 2),
            "MSE": round(mse, 2)
        }

class ClassificationEvaluation(ModelEvaluationStrategy):
    @staticmethod
    def evaluate_model(model, X_test, y_test)->dict:
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred, output_dict=True)


class RecommendationEvaluation(ModelEvaluationStrategy):
    @staticmethod
    def evaluate_model(model_dict: dict, X_test: pd.DataFrame, y_test: Any = None, K: int = 10) -> dict:
        user_item_matrix = model_dict["user_item_matrix"]
        cosine_sim = model_dict["cosine_similarity"]
        product_keys = model_dict["product_keys"]

        hits = 0
        total = 0

        for user_id in X_test.index:
            user_vector = X_test.loc[user_id].values

            if user_vector.sum() == 0:
                continue 

            ground_truth_index = np.argmax(user_vector)
            ground_truth_product = X_test.columns[ground_truth_index]

            seen_indices = np.where(user_vector > 0)[0]
            unseen_indices = list(set(range(len(product_keys))) - set(seen_indices))

            user_scores = cosine_sim[ground_truth_index, unseen_indices]
            top_k_indices = np.argsort(user_scores)[-K:][::-1]
            recommended_products = [product_keys[unseen_indices[i]] for i in top_k_indices]

            if ground_truth_product in recommended_products:
                hits += 1

            total += 1

        recall_at_k = hits / total if total > 0 else 0

        return {
            "Recall_at_K": round(recall_at_k, 4),
            "Users Evaluated": total,
            "Hit Count": hits,
            "K": K
        }
