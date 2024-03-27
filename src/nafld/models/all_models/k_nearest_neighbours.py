from nafld.models.abstract_model import AbstractModel
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier


class KNNModel(AbstractModel):
    def create_new_model(self) -> BaseEstimator:
        self.model = KNeighborsClassifier(**self.MODELS[self.name]["best_hyper_params"])
