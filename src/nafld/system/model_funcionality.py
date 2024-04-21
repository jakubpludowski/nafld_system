import pandas as pd
from nafld.config.base_config import ConfigBase
from nafld.models.abstract_model import AbstractModel
from nafld.models.all_models.ensemble import EnsembleModel
from pandas import DataFrame
from runscripts.manage_data.configs.step_1_4_config import MODELS_NAMES_TO_RENAME, THRESHOLD_TO_BE_PART_OF_ENSEMBLE
from sklearn.metrics import f1_score


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


def train_all_models(all_models: list[AbstractModel], data: DataFrame, feature_names: list[str]) -> list[AbstractModel]:
    for model in all_models:
        model.train_model(data, feature_names)

    return all_models


def validate_all_models(
    run_details: DataFrame, all_models: list[AbstractModel], data: DataFrame
) -> list[AbstractModel]:
    for model in all_models:
        basic_stats, auc_values, predictions = model.validate_model(data)
        run_details.loc[run_details["ModelName"] == model.name, "F1New"] = basic_stats["f1"]
        run_details.loc[run_details["ModelName"] == model.name, "Accuracy"] = basic_stats["accuracy"]
        run_details.loc[run_details["ModelName"] == model.name, "Precision"] = basic_stats["precision"]
        run_details.loc[run_details["ModelName"] == model.name, "Recall"] = basic_stats["recall"]
        run_details.loc[run_details["ModelName"] == model.name, "Roc Auc"] = auc_values["roc_auc"]

    return run_details, all_models


def overwrite_models(run_details: DataFrame, all_models: list[AbstractModel]) -> list[AbstractModel]:
    for model in all_models:
        new_model_value = run_details.loc[run_details["ModelName"] == model.name, "F1New"].values[0]
        old_model_value = run_details.loc[run_details["ModelName"] == model.name, "F1"].values[0]

        if new_model_value > old_model_value:
            model.f1 = new_model_value
            model.save_to_file()

    run_details["Part of Ensemble"] = run_details["Roc Auc"] > THRESHOLD_TO_BE_PART_OF_ENSEMBLE

    return run_details, all_models


def tidy_run_details(run_details: DataFrame) -> DataFrame:
    run_details["F1"] = run_details.apply(lambda row: max(row["F1"], row["F1New"]), axis=1)

    run_details.drop(columns=["F1New", "IsModelNew"], inplace=True)
    run_details["ModelName"] = run_details["ModelName"].map(MODELS_NAMES_TO_RENAME)
    return run_details


def test_ensemble_model(
    run_details: DataFrame, ensemble_model: EnsembleModel, all_models: list[AbstractModel], data: DataFrame
) -> float:
    model_predictions = pd.DataFrame()
    f1_metrics = []
    (X_train, y_train, X_test, y_test) = data
    for model in all_models:
        preds = model.make_predictions(X_test)
        model_predictions[f"{model.name}"] = preds
        f1_metrics.append(f1_score(preds, y_test))

    model_predictions["mean_preds"] = model_predictions.mean(axis=1)

    mean_f1_metric_for_all_models = sum(f1_metrics) / len(f1_metrics)

    models_to_ensemble = [
        (model.name, model.model)
        for model in all_models
        if (run_details.loc[run_details["ModelName"] == model.name, "Part of Ensemble"].iloc[0])
    ]
    ensemble_model.create_new_model(models=models_to_ensemble)
    ensemble_model.model.fit(X_train, y_train)
    preds = ensemble_model.make_predictions(X_test)

    basic_ensemble_stats, ensemble_auc_values, _ = ensemble_model.validate_model(data=data)
    ensemble_model.save_to_file()
    model_predictions["ensemble"] = preds
    model_predictions["label"] = y_test.reset_index(drop=True)

    return model_predictions, basic_ensemble_stats, ensemble_auc_values, mean_f1_metric_for_all_models
