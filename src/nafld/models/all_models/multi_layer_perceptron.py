from nafld.models.abstract_model import AbstractModel
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier


class MLPModel(AbstractModel):
    def create_new_model(self) -> BaseEstimator:
        self.model = MLPClassifier(**self.MODELS[self.name]["best_hyper_params"])
