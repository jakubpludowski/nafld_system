from nafld.table.tables.features import BaseFeaturesTable
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment

if __name__ == "__main__":
    CONF, GLOBAL_MODELS = initialize_environment()

    raw_features_table = StaticTable(name="raw_features", path_to_table=CONF.DATA_RAW)
    raw_inputs_features_table = StaticTable(name="raw_inputs_features", path_to_table=CONF.INPUTS_PROCESSED_DATA)
    base_database_features_table = StaticTable(name="base_database_features", path_to_table=CONF.DATA_BASE)

    base_features_table = StaticTable(name="base_features", path_to_table=CONF.DATA_BASE)
    new_base_features_table = StaticTable(name="new_base_features", path_to_table=CONF.NEW_DATA_BASE)

    base_features_table = BaseFeaturesTable(base_features_table)
    base_features_table.generate(
        raw_features_table, raw_inputs_features_table, base_database_features_table, new_base_features_table
    )
