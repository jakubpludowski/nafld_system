import uuid

import pandas as pd
from nafld.table.processed_table import ProcessedPatientFeaturesColumns
from nafld.table.tables.base_table import BaseTable
from nafld.table.tables.static_table import StaticTable
from pandas import DataFrame


class BaseFeaturesTable(BaseTable):
    def generate(
        self,
        features_raw_table: StaticTable,
        features_raw_inputs_table: StaticTable,
        features_from_database_table: StaticTable,
        new_base_features_table: StaticTable,
    ) -> None:
        features_raw = features_raw_table.read(file_format="csv")
        features_raw_inputs = features_raw_inputs_table.read(file_format="csv")
        features_from_database = features_from_database_table.read(file_format="parquet")

        data_to_concat = [df for df in [features_raw, features_raw_inputs, features_from_database] if df is not None]
        if data_to_concat:
            for df in data_to_concat:
                if df.isnull().values.any():
                    raise ValueError(f"There are missing values in data {df.name}")

            features_base = pd.concat(data_to_concat, ignore_index=True)

            features_base.drop_duplicates(inplace=True)
            features_base = self.add_unique_id_for_patients(features_base)

            base_features_increment_table = BaseFeaturesTable(new_base_features_table)
            if features_from_database is not None:
                new_data = pd.merge(features_base, features_from_database, how="left", indicator=True)
                new_data = new_data[new_data["_merge"] == "left_only"].drop("_merge", axis=1)
                base_features_increment_table.table.write_parquet(df=new_data)
            else:
                base_features_increment_table.table.write_parquet(df=features_base)
        else:
            raise ValueError("No data found")

        if features_base.isnull().sum().sum() == 0:
            self.table.write_parquet(df=features_base)
            if features_raw_inputs is not None:
                features_raw_inputs_table.delete_file()
        else:
            raise ValueError("There are missing value in the final dataset")

    def add_unique_id_for_patients(self, df: DataFrame) -> DataFrame:
        rows_with_null = df[df[ProcessedPatientFeaturesColumns.PatiendId].isnull()]
        if rows_with_null.empty:
            return df

        unique_ids = [uuid.uuid4() for _ in range(len(rows_with_null))]

        df.loc[rows_with_null.index, ProcessedPatientFeaturesColumns.PatiendId] = unique_ids
        df[ProcessedPatientFeaturesColumns.PatiendId] = df[ProcessedPatientFeaturesColumns.PatiendId].astype(str)
        return df
