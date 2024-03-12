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

        # TODO: add all necessary filtering
        features_base = features_raw
        self.table.write_parquet(df=features_base)
