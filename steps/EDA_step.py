from typing import Dict
import pandas as pd
from src.EDA import YDataProfilingEDA
from zenml import step

@step
def eda_step(data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    eda = YDataProfilingEDA.generate_report(data)
    return eda
