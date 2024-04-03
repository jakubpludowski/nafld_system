import uuid

import pandas as pd
from nafld.table.processed_table import ProcessedPatientFeaturesColumns
from nafld.table.tables.base_table import BaseTable
from nafld.table.tables.static_table import StaticTable
from pandas import DataFrame


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

            columns_to_drop_duplicates = ProcessedPatientFeaturesColumns.get_table_columns()
            columns_to_drop_duplicates.remove(ProcessedPatientFeaturesColumns.PatiendId)

            features_base.drop_duplicates(subset=columns_to_drop_duplicates, inplace=True)

            features_base = self.add_unique_id_for_patients(features_base)

        else:
            raise ValueError("No data found")

        if features_base.isnull().sum().sum() == 0:
            self.table.write_parquet(df=features_base)
        else:
            raise ValueError("There are missing value in the final dataset")

    def add_unique_id_for_patients(self, df: DataFrame) -> DataFrame:
        unique_ids = [uuid.uuid4() for _ in range(len(df))]
        df[ProcessedPatientFeaturesColumns.PatiendId] = unique_ids
        df[ProcessedPatientFeaturesColumns.PatiendId] = df[ProcessedPatientFeaturesColumns.PatiendId].astype(str)
        return df
