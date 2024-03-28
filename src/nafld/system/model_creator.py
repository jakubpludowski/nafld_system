from nafld.models.configs.hyper_parameter_config import MODELS_OBJECTS, MODELS_TO_TRAIN
from pandas import DataFrame


def train_all_models(data: DataFrame, global_models: dict, path_to_models: str, path_to_best_params: str) -> None:
    for model_name in MODELS_TO_TRAIN:
        model = MODELS_OBJECTS[model_name](model_name, global_models, path_to_models, path_to_best_params)
        model.load_model(warm_start=True)
        model.train_model(data)
        model.save_to_file()


def tune_hyper_params_for_all_models(
    data: DataFrame, global_models: dict, path_to_models: str, path_to_best_params: str
) -> None:
    for model_name in MODELS_TO_TRAIN:
        model = MODELS_OBJECTS[model_name](model_name, global_models, path_to_models, path_to_best_params)
        model.load_model(warm_start=True)
        model.get_hyper_parameters(data)
