from nafld.config.base_config import ConfigBase
from nafld.models.abstract_model import AbstractModel
from pandas import DataFrame


def load_all_models(
    run_details: DataFrame, data: DataFrame, all_models: list[AbstractModel], config: ConfigBase
) -> tuple[DataFrame, list[AbstractModel]]:
    tune_huperparams = config.tune_hyperparams

    for model in all_models:
        (name, is_new, f1) = model.load_model()
        run_details.loc[run_details["ModelName"] == name, "IsModelNew"] = is_new
        run_details.loc[run_details["ModelName"] == name, "F1"] = f1
        if is_new:
            model.train_model(data)
            if tune_huperparams:
                model.get_hyper_parameters(data)
        model.save_to_file()

    return run_details, all_models


def set_model_to_warm_start(all_models: list[AbstractModel]) -> list[AbstractModel]:
    for model in all_models:
        model.change_warm_start()

    return all_models


def train_all_models(all_models: list[AbstractModel], data: DataFrame) -> list[AbstractModel]:
    for model in all_models:
        model.train_model(data)

    return all_models
