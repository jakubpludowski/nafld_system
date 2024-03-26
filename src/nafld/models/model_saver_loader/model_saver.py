import pickle
from pathlib import Path

from nafld.models.model_saver_loader.models_config import MODELS
from sklearn.base import BaseEstimator


class Model:
    def __init__(self, path_to_models: str) -> None:
        self.path_to_models = path_to_models

    def load_model(self, name: str) -> BaseEstimator:
        path = Path(self.path_to_models) / MODELS[name]["path"]
        return pickle.loads(path)  # noqa: S301
