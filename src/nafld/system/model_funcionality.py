from nafld.config.base_config import ConfigBase
from nafld.models.abstract_model import AbstractModel
from pandas import DataFrame


def load_all_models(
    run_details: DataFrame, data: DataFrame, all_models: list[AbstractModel], config: ConfigBase
) -> tuple[DataFrame, list[AbstractModel]]:
    for model in all_models:
        (name, is_new, f1) = model.load_model()
        run_details.loc[run_details["ModelName"] == name, "IsModelNew"] = is_new
        run_details.loc[run_details["ModelName"] == name, "F1"] = f1
        if is_new:
            if config.tune_hyperparams:
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


def validate_all_models(
    run_details: DataFrame, all_models: list[AbstractModel], data: DataFrame
) -> list[AbstractModel]:
    for model in all_models:
        basic_stats, auc_values, predictions = model.validate_model(data)
        run_details.loc[run_details["ModelName"] == model.name, "F1New"] = basic_stats["f1"]

    return run_details, all_models


def overwrite_models(run_details: DataFrame, all_models: list[AbstractModel]) -> list[AbstractModel]:
    for model in all_models:
        new_model_value = run_details.loc[run_details["ModelName"] == model.name, "F1New"].values[0]
        old_model_value = run_details.loc[run_details["ModelName"] == model.name, "F1"].values[0]

        if new_model_value > old_model_value:
            model.f1 = new_model_value
            model.save_to_file()

    return run_details, all_models
