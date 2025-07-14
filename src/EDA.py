from abc import ABC, abstractmethod
import pandas as pd
import os
from ydata_profiling import ProfileReport

class BaseEDA(ABC):
    @abstractmethod
    def generate_report(self, data: dict) -> dict[str, str]:
        pass


class YDataProfilingEDA(BaseEDA):
    @staticmethod
    def generate_report(data: dict[str, pd.DataFrame]) -> dict[str, str]:
        report_paths = {}
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        for table_name, df in data.items():
            report_path = os.path.join(output_dir, f"{table_name.replace('.', '_')}_eda.html")
            profile = ProfileReport(df, title=f"EDA Report: {table_name}", explorative=True)
            profile.to_file(report_path)
            report_paths[table_name] = report_path
        
        return report_paths

    