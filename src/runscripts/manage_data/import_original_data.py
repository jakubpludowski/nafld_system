from nafld.table.tables.features import BaseFeaturesTable
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment

if __name__ == "__main__":
    CONF = initialize_environment()

    raw_features_table = StaticTable(mode=CONF.MODE, data_level="raw", name="features")
    base_features_table = StaticTable(mode=CONF.MODE, data_level="base", name="features")

    base_features_table = BaseFeaturesTable(base_features_table)
    base_features_table.generate(raw_features_table, CONF.RAW_FEATURES_TABLE_PARQUET, CONF.BASE_FEATURES_TABLE_PARQUET)

    # convert_csv_to_parquet(CONF.ORIGINAL_DATA_CSV_FILE, CONF.RAW_FEATURES_TABLE_PARQUET)  # noqa: ERA001
