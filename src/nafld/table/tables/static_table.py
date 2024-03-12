import pandas as pd
from nafld.table.tables.abstract_table import AbstractTable
from pandas import DataFrame


class StaticTable(AbstractTable):
    def __init__(self, name: str, path_to_table: str) -> None:
        self.name = name
        self.path = path_to_table

    def read(self, file_format: str = "parquet") -> DataFrame:
        if file_format == "parquet":
            try:
                df = pd.read_parquet(self.path)
            except FileNotFoundError:
                return None
        elif file_format == "csv":
            try:
                df = pd.read_csv(self.path)
            except FileNotFoundError:
                return None
        return df

    def write_parquet(self, df: DataFrame) -> None:
        df.to_parquet(self.path)

    def write_csv(self, df: DataFrame) -> None:
        df.to_csv(self.path)
