from abc import ABC, abstractmethod
import pandas as pd
import pyodbc
from typing import Dict, List

class DataIngestor(ABC):
    @abstractmethod
    def ingest_data(self) -> dict:
        pass

class SQLServerIngestor(DataIngestor):
    def __init__(self, connection_string: str, tables: List[str]):
        self.connection_string = connection_string
        self.tables = tables

    def ingest_data(self) -> Dict[str, pd.DataFrame]:
        try:
            print(f"Attempting to connect with: {self.connection_string}")
            conn = pyodbc.connect(self.connection_string)
            print("Connection successful!")
            data = {}
            for table in self.tables:
                print(f"Reading table: {table}")
                df = pd.read_sql(f"SELECT * FROM {table}", conn)
                data[table] = df
            conn.close()
            new_names = ['customers', 'products', 'sales']
            data = dict(zip(new_names, data.values()))
            return data
        except pyodbc.OperationalError as e:
            print(f"Connection failed: {e}")
            raise


class ExcelIngestor(DataIngestor):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def ingest_data(self) -> dict[str, pd.DataFrame]:
        excel_file = pd.ExcelFile(self.file_path)
        sheets = excel_file.sheet_names[:3]
        return {sheet: pd.read_excel(self.file_path, sheet_name=sheet) for sheet in sheets}

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(source_type: str, connection_string: str = "", tables: List[str] = None, file_path: str = "") -> DataIngestor:
        if source_type == "sql":
            if not connection_string or not tables:
                raise ValueError("connection_string and tables are required for SQL ingestion.")
            return SQLServerIngestor(connection_string, tables)
        elif source_type == "excel":
            if not file_path:
                raise ValueError("file_path is required for Excel ingestion.")
            return ExcelIngestor(file_path)
        else:
            raise ValueError(f"Unsupported input type: {source_type}")

# if __name__ == '__main__':
#     load_dotenv(".env")
#     DRIVER = os.getenv("DRIVER")
#     SERVER = os.getenv("SERVER")
#     DATABASE = os.getenv("DATABASE")
#     TRUST_CONNECTION = os.getenv("TRUST_CONNECTION", "true")
#     TABLE_NAMES = os.getenv("TABLE_NAMES", "").split(",")
#     trusted_conn_value = "yes" if TRUST_CONNECTION.lower() == "true" else "no"
    
#     CONNECTION_STRING = (
#         f"DRIVER={DRIVER};"
#         f"SERVER={SERVER};"
#         f"DATABASE={DATABASE};"
#         f"Trusted_Connection={trusted_conn_value}"
#     )

#     sql_ingestor = SQLServerIngestor(CONNECTION_STRING, TABLE_NAMES)
#     sql_data = sql_ingestor.ingest_data()
#     for table, df in sql_data.items():
#         print(f"Table: {table}, Rows: {len(df)}")


