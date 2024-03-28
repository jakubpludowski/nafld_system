from nafld.models.data_preparation import prepare_data
from nafld.system.model_creator import train_all_models
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment

if __name__ == "__main__":
    CONF, GLOBAL_MODELS = initialize_environment()

    base_features_table = StaticTable(name="base_features", path_to_table=CONF.DATA_BASE)

    preprocessed_data = prepare_data(base_features_table.read())

    train_all_models(preprocessed_data, GLOBAL_MODELS, CONF.DATA_MODELS_DIRECTORY, CONF.PATH_TO_BEST_PARAMETERS)
