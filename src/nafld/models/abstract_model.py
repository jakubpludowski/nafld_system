import copy
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

from nafld.models.configs.models_config import CV_FOR_RANDOM_SEARCH, N_ITER, RANDOM_STATE, TEST_SIZE
from nafld.utils.model_utils.best_parameters_loader import save_best_parameters
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV


class AbstractModel(ABC):
    model: BaseEstimator

    def __init__(
        self, name: str, global_models: dict, path_to_models: str, path_to_best_params: str, warm_start: bool = False
    ) -> None:
        self.name = name
        self.folder_name = path_to_models
        self.MODELS = global_models
        self.path = Path(path_to_models) / (self.name + ".pkl")
        self.path_to_best_parameters = path_to_best_params
        self.random_state = RANDOM_STATE
        self.test_size = TEST_SIZE
        self.warm_start = warm_start
        self.f1 = 0

    def load_model_from_file(self) -> None:
        with Path.open(Path(self.path), "rb") as file:
            self.model, self.f1 = pickle.load(file)  # noqa: S301

    def save_to_file(self) -> None:
        with Path.open(Path(self.path), "wb") as file:
            model_with_f1 = (self.model, self.f1)
            pickle.dump(model_with_f1, file)

    def check_if_model_is_saved(self) -> bool:
        file_name = Path(self.path).name
        folder_contents = [entry.name for entry in Path(self.folder_name).iterdir() if entry.is_file()]
        return file_name in folder_contents

    def load_model(self) -> tuple[str, bool]:
        if self.check_if_model_is_saved():
            self.load_model_from_file()
            return self.name, 0, self.f1
        self.create_new_model()
        return self.name, 1, 0

    def save_new_best_parameters(self, model_dict: dict) -> None:
        save_best_parameters(self.path_to_best_parameters, model_dict)

    def train_model(self, data: tuple, feature_names: list[str]) -> None:
        (X_train, y_train, _, _) = data
        if feature_names is None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
            self.model.feature_names = feature_names

    def validate_model(self, data: tuple) -> tuple[dict, dict, dict]:
        (_, _, X_test, y_test) = data
        predictions = self.make_predictions(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        auc_roc = roc_auc_score(y_test, predictions).round(decimals=2)
        pr, re, _ = precision_recall_curve(y_test, predictions)
        auc_pr = auc(re, pr)

        basic_stats = {"f1": f1, "accuracy": acc, "precision": precision, "recall": recall}

        auc_values = {"roc_auc": auc_roc, "pr_auc": auc_pr, "confusion_matrix": cm}

        predictions_and_y_test = {"predictions": predictions, "label": y_test}

        return basic_stats, auc_values, predictions_and_y_test

    def make_predictions(self, test_data_X: DataFrame) -> DataFrame:
        return self.model.predict(test_data_X)

    def get_hyper_parameters(self, data: tuple) -> dict:
        (X_train, y_train, X_test, y_test) = data

        # Find new hyper parameters
        param_dist = self.MODELS[self.name]["search_hyper_params"]
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            n_iter=N_ITER,
            cv=CV_FOR_RANDOM_SEARCH,
            random_state=RANDOM_STATE,
        )
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_

        # Check new model accuracy
        model_with_new_params = copy.deepcopy(self.model).set_params(**best_params)
        model_with_new_params.fit(X_train, y_train)
        y_pred = model_with_new_params.predict(X_test)
        new_model_score = f1_score(y_test, y_pred)

        # Decide if new hyper parameters should replace old ones
        if new_model_score >= self.f1:
            self.MODELS[self.name]["best_hyper_params"] = best_params
            self.f1 = new_model_score
        else:
            self.MODELS[self.name]["best_hyper_params"] = self.model.get_params()
        self.save_new_best_parameters(self.MODELS)

    def change_warm_start(self) -> None:
        if hasattr(self.model, "warm_start"):
            self.model.warm_start = self.warm_start
        else:
            print(f"Model {self.name} does not provide warm start")  # noqa: T201

    @abstractmethod
    def create_new_model() -> BaseEstimator:
        raise NotImplementedError
