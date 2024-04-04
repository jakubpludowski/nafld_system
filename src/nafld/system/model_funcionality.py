import pandas as pd
from nafld.config.base_config import ConfigBase
from nafld.models.abstract_model import AbstractModel
from pandas import DataFrame, Series
from sklearn.metrics import f1_score, roc_auc_score


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


def test_ensemble_model(all_models: list[AbstractModel], data: DataFrame) -> float:
    model_predictions = pd.DataFrame()
    f1_metrics = []
    (_, _, X_test, y_test) = data
    for model in all_models:
        preds = model.make_predictions(X_test)
        model_predictions[f"{model.name}"] = preds
        f1_metrics.append(f1_score(preds, y_test))

    ensemble_preds = model_predictions.mean(axis=1)
    return roc_auc_score(y_test, ensemble_preds), sum(f1_metrics) / len(f1_metrics)


def predict_with_ensemble_model(all_models: list[AbstractModel], X_test: DataFrame) -> Series:
    model_predictions = pd.DataFrame()
    for model in all_models:
        preds = model.make_predictions(X_test)
        model_predictions[f"{model.name}"] = preds

    return model_predictions.mean(axis=1)
