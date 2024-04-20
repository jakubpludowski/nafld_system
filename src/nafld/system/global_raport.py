import dalex as dx
from nafld.models.all_models.ensemble import EnsembleModel
from pandas import DataFrame


def generate_global_raport(model: EnsembleModel, data: DataFrame) -> None:
    perform_shap_analysis(model=model, data=data)


def perform_shap_analysis(model: EnsembleModel, data: DataFrame) -> None:
    (X_train, y_train, X_test, y_test) = data

    exp = dx.Explainer(model.model, X_train, y_train)
    exp.model_performance(model_type="classification")
