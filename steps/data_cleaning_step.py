from typing import Tuple, Annotated
import pandas as pd
from zenml import step
from src.data_cleaning import CustomersFormator, ProductFormator, SalesFormator

@step
def data_cleaning_step(
    data: dict[str, pd.DataFrame]
) -> Tuple[
    Annotated[pd.DataFrame, "cleaned_customers"],
    Annotated[pd.DataFrame, "cleaned_products"],
    Annotated[pd.DataFrame, "cleaned_sales"]
]:
    customers, products, sales = data.values()

    cleaned_customers = CustomersFormator().transform(customers)
    cleaned_products = ProductFormator().transform(products)
    cleaned_sales = SalesFormator().transform(sales)

    return cleaned_customers, cleaned_products, cleaned_sales
