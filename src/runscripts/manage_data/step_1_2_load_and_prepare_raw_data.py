from nafld.system.original_data_loader import DataLoader
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment
from runscripts.manage_data.configs.step_1_2_config import SEED

if __name__ == "__main__":
    CONF = initialize_environment()

    input_original_features_table = StaticTable(name="original_features", path_to_table=CONF.INPUTS_ORIGINAL_DATA)
    output_raw_features_table = StaticTable(name="raw_features", path_to_table=CONF.INPUTS_PROCESSED_DATA)

    original_data_loader = DataLoader(
        input_original_features_table, output_raw_features_table, mode="binary", seed=SEED
    )

    original_data_loader.process_data()
