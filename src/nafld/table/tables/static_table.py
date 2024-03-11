import pandas as pd
from nafld.table.tables.abstract_table import AbstractTable
from pandas import DataFrame


class StaticTable(AbstractTable):
    def __init__(self, mode: str, data_level: str, name: str) -> None:
        self.mode = mode
        self.data_level = data_level
        self.name = name

    def read(self, file_path: str, file_format: str = "parquet") -> DataFrame:
        if file_format == "parquet":
            df = pd.read_parquet(file_path)
        elif file_format == "csv":
            df = pd.read_csv(file_path)
        return df

    def write_parquet(self, df: DataFrame, path: str) -> DataFrame:
        df.to_parquet(path)
