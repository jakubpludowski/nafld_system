from nafld.table.tables.base_table import BaseTable
from nafld.table.tables.static_table import StaticTable


class BaseFeaturesTable(BaseTable):
    table: StaticTable

    def generate(self, features_raw: StaticTable, path_raw: str, path_base: str) -> None:
        features_raw = features_raw.read(path_raw, file_format="csv")

        # TODO: add all necessary filtering
        features_base = features_raw
        self.table.write_parquet(df=features_base, path=path_base)
