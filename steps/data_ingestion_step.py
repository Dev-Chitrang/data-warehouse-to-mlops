import os
from typing import Dict, Optional
import pandas as pd
from dotenv import load_dotenv
from zenml import step
from src.ingest_data import DataIngestorFactory

load_dotenv(".env")

@step
def data_ingestion_step(file_type: str, file_path: Optional[str] = None) -> dict:
    if file_type == "sql":
        conn_string = (
            f"DRIVER={os.getenv('DRIVER')};"
            f"SERVER={os.getenv('SERVER')};"
            f"DATABASE={os.getenv('DATABASE')};"
            f"Trusted_Connection={os.getenv('TRUSTED_CONNECTION')};"
        )
        tables = os.getenv("TABLE_NAMES", "").split(",")

        data_ingestor = DataIngestorFactory.get_data_ingestor(
            source_type="sql",
            connection_string=conn_string,
            tables=tables
        )

        return data_ingestor.ingest_data()

    elif file_type == "excel":
        if not file_path:
            raise ValueError("file_path is required when file_type is 'excel'")

        data_ingestor = DataIngestorFactory.get_data_ingestor(
            source_type="excel",
            file_path=file_path
        )

        return data_ingestor.ingest_data()

    else:
        raise ValueError(f"Unsupported file_type '{file_type}'. Use 'sql' or 'excel'.")
