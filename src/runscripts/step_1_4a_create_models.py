from nafld.models.all_models.random_forest import RandomForestModel
from nafld.models.data_preparation import prepare_data
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment

if __name__ == "__main__":
    CONF, GLOBAL_MODELS = initialize_environment()

    base_features_table = StaticTable(name="base_features", path_to_table=CONF.DATA_BASE)

    preprocessed_data = prepare_data(base_features_table.read())

    rf_model = RandomForestModel(
        "random_forest", GLOBAL_MODELS, CONF.DATA_MODELS_DIRECTORY, CONF.PATH_TO_BEST_PARAMETERS
    )
    rf_model.load_model(warm_start=True)
    rf_model.train_model(preprocessed_data)
    rf_model.get_hyper_parameters(preprocessed_data)
    rf_model.save_to_file()
