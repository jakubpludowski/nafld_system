import pickle
from abc import ABC, abstractmethod
from pathlib import Path

from nafld.models.configs.models_config import RANDOM_STATE, TEST_SIZE
from nafld.utils.model_utils.best_parameters_loader import save_best_parameters
from pandas import DataFrame
from sklearn.base import BaseEstimator


class AbstractModel(ABC):
    model: BaseEstimator

    def __init__(self, name: str, global_models: dict, path_to_models: str, path_to_best_params: str) -> None:
        self.name = name
        self.folder_name = path_to_models
        self.MODELS = global_models
        self.path = Path(path_to_models) / (self.name + ".pkl")
        self.path_to_best_parameters = path_to_best_params
        self.random_state = RANDOM_STATE
        self.test_size = TEST_SIZE

    def load_model_from_file(self) -> None:
        with Path.open(Path(self.path), "rb") as file:
            self.model = pickle.load(file)  # noqa: S301

    def save_to_file(self) -> None:
        with Path.open(Path(self.path), "wb") as file:
            pickle.dump(self.model, file)

    def check_if_model_is_saved(self) -> bool:
        file_name = Path(self.path).name
        folder_contents = [entry.name for entry in Path(self.folder_name).iterdir() if entry.is_file()]
        return file_name in folder_contents

    def load_model(self, warm_start: bool) -> None:
        if warm_start:
            if self.check_if_model_is_saved():
                self.load_model_from_file()
            else:
                self.create_new_model()
        else:
            self.create_new_model()

    def save_new_best_parameters(self, model_dict: dict) -> None:
        save_best_parameters(self.path_to_best_parameters, model_dict)

    def train_model(self, data: tuple) -> None:
        (X_train, y_train, _, _) = data
        self.model.fit(X_train, y_train)

    def make_predictions(self, test_data_X: DataFrame) -> DataFrame:
        return self.model.predict(test_data_X)

    @abstractmethod
    def get_hyper_parameters() -> dict:
        raise NotImplementedError

    @abstractmethod
    def create_new_model() -> BaseEstimator:
        raise NotImplementedError
