from nafld.table.tables.features import BaseFeaturesTable
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment

if __name__ == "__main__":
    CONF = initialize_environment()

    raw_features_table = StaticTable(name="raw_features", path_to_table=CONF.INPUTS_NEW_DATA)
    raw_inputs_features_table = StaticTable(name="raw_inputs_features", path_to_table=CONF.INPUTS_PROCESSED_DATA)
    base_features_table = StaticTable(name="base_features", path_to_table=CONF.DATA_BASE)
    base_database_features_table = StaticTable(name="base_database_features", path_to_table=CONF.DATA_BASE)

    base_features_table = BaseFeaturesTable(base_features_table)
    base_features_table.generate(raw_features_table, raw_inputs_features_table, base_database_features_table)

    # convert_csv_to_parquet(CONF.ORIGINAL_DATA_CSV_FILE, CONF.RAW_FEATURES_TABLE_PARQUET)  # noqa: ERA001
