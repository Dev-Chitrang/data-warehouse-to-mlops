from zenml import step
import pandas as pd
import random
from datetime import datetime, timedelta

@step
def dynamic_loader(model_type: str):
    if model_type == "classification":
        return pd.DataFrame([{
            "customer_key": random.randint(1000, 9999),
            "Accessories": random.randint(0, 10),
            "Bikes": random.randint(0, 5),
            "Clothing": random.randint(0, 8),
            "country_Canada": random.choice([True, False]),
            "country_France": random.choice([True, False]),
            "country_Germany": random.choice([True, False]),
            "country_United Kingdom": random.choice([True, False]),
            "country_United States": random.choice([True, False])
        }])
    
    elif model_type == "forecasting":
        date = datetime(2012, 1, 1) + timedelta(days=random.randint(0, 1000))
        return pd.DataFrame([{"ds": date.strftime("%Y-%m-%d")}])

    elif model_type == "recommendation":
        customer_key = random.randint(1000, 9999)
        product_ids = random.sample(range(3, 296), k=10)
        values = [round(random.uniform(1, 50), 1) for _ in product_ids]
        data = {str(pid): val for pid, val in zip(product_ids, values)}
        data["customer_key"] = customer_key
        return list(data.values())

    else:
        raise ValueError("Invalid model type.")
