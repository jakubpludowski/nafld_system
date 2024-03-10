import pandas as pd


def convert_csv_to_parquet(path_csv: str, path_parquet: str) -> None:
    pd.read_csv(path_csv).to_parquet(path_parquet)
