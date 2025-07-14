from abc import ABC, abstractmethod
import pandas as pd
from typing import Union

class FeatureEngineer(ABC):
    @abstractmethod
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class SalesForecastingFeatureEngineer(FeatureEngineer):
    def engineer_features(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = data["sales"].copy()
        df = df[["order_date", "sales_amount"]]
        df = df.groupby("order_date", as_index=False).sum()
        df.rename(columns={"order_date": "ds", "sales_amount": "y"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")
        return df

class CategoryClassificationEngineer(FeatureEngineer):
    def engineer_features(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sales = data["sales"]
        customers = data["customers"]
        products = data["products"]

        df = sales.merge(customers, on="customer_key", how="left") \
                  .merge(products, on="product_key", how="left")

        cat_counts = df.groupby(["customer_key", "category"]).size() \
                       .unstack(fill_value=0).reset_index()

        cat_counts["label"] = cat_counts.drop(columns="customer_key").idxmax(axis=1)
        cat_counts["label"] = pd.Categorical(cat_counts["label"]).codes

        cust_info = customers[["customer_key", "country"]]
        df_final = cat_counts.merge(cust_info, on="customer_key")
        df_final = pd.get_dummies(df_final, columns=["country"] ,drop_first=True)
        df_final.dropna(inplace=True)

        return df_final


class RecommendationFeatureEngineer(FeatureEngineer):
    def engineer_features(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sales = data["sales"]

        if sales.empty:
            raise ValueError("Sales data is empty. Cannot engineer recommendation features.")


        user_item = sales.pivot_table(
            index="customer_key",
            columns="product_key",
            values="sales_amount",
            aggfunc="sum"
        ).fillna(0).sort_index()

        return user_item