import pandas as pd
from nafld.table.processed_table import ProcessedPatientFeaturesColumns
from nafld.table.tables.base_table import BaseTable
from nafld.table.tables.static_table import StaticTable


class BaseFeaturesTable(BaseTable):
    def generate(
        self,
        features_raw: StaticTable,
        features_raw_inputs: StaticTable,
        features_from_database: StaticTable,
    ) -> None:
        features_raw = features_raw.read(file_format="csv")
        features_raw_inputs = features_raw_inputs.read(file_format="csv")
        features_from_database = features_from_database.read(file_format="parquet")

        data_to_concat = [df for df in [features_raw, features_raw_inputs, features_from_database] if df is not None]
        if data_to_concat:
            for df in data_to_concat:
                if df.isnull().values.any():
                    raise ValueError(f"There are missing values in data {df.name}")

            features_base = pd.concat(data_to_concat, ignore_index=True)
            features_base.drop_duplicates(subset=ProcessedPatientFeaturesColumns.PatiendId, inplace=True)

        else:
            raise ValueError("No data found")

        if features_base.isnull().sum().sum() == 0:
            self.table.write_parquet(df=features_base)
        else:
            raise ValueError("There are missing value in the final dataset")
