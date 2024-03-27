from nafld.models.all_models.ada_boost import AdaBoostModel
from nafld.models.all_models.logistic_regression import LogisticRegressionModel
from nafld.models.all_models.multi_layer_perceptron import MLPModel
from nafld.models.data_preparation import prepare_data
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment

if __name__ == "__main__":
    CONF, GLOBAL_MODELS = initialize_environment()

    base_features_table = StaticTable(name="base_features", path_to_table=CONF.DATA_BASE)

    preprocessed_data = prepare_data(base_features_table.read())

    log_model = LogisticRegressionModel(
        "log_reg", GLOBAL_MODELS, CONF.DATA_MODELS_DIRECTORY, CONF.PATH_TO_BEST_PARAMETERS
    )
    log_model.load_model(warm_start=True)
    log_model.train_model(preprocessed_data)
    log_model.get_hyper_parameters(preprocessed_data)
    log_model.save_to_file()

    ada_model = AdaBoostModel("adaboost", GLOBAL_MODELS, CONF.DATA_MODELS_DIRECTORY, CONF.PATH_TO_BEST_PARAMETERS)
    ada_model.load_model(warm_start=True)
    ada_model.train_model(preprocessed_data)
    ada_model.get_hyper_parameters(preprocessed_data)
    ada_model.save_to_file()

    mlp_model = MLPModel("mlp", GLOBAL_MODELS, CONF.DATA_MODELS_DIRECTORY, CONF.PATH_TO_BEST_PARAMETERS)
    mlp_model.load_model(warm_start=True)
    mlp_model.train_model(preprocessed_data)
    mlp_model.get_hyper_parameters(preprocessed_data)
    mlp_model.save_to_file()
