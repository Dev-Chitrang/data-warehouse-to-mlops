from abc import ABC, abstractmethod
import pandas as pd

class DataFormater(ABC):
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class CustomersFormator(DataFormater):
    def transform(self, data) -> pd.DataFrame:
        df = data.copy()
        df['gender'].fillna(df['gender'].mode()[0], inplace=True)
        df['country'].fillna(df['country'].mode()[0], inplace=True)

        df['gender'] = df['gender'].astype('category')
        df['marital_status'] = df['marital_status'].astype('category')
        df['country'] = df['country'].astype('category')

        return df

class ProductFormator(DataFormater):
    def transform(self, data: pd.DataFrame):
        df = data.copy()
        for col in ['category', 'subcategory', 'product_line', 'maintenance']:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        if 'cost' in df.columns:
            df['cost_bin'] = pd.qcut(df['cost'], q=4, labels=False, duplicates='drop')

        return df

class SalesFormator(DataFormater):
    def transform(self, data: pd.DataFrame):
        df = data.copy()
        df.dropna(subset=['order_date', 'sales_amount'], inplace=True)
        df.drop_duplicates(inplace=True)

        return df
