from nafld.models.abstract_model import AbstractModel
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier


class AdaBoostModel(AbstractModel):
    def create_new_model(self) -> BaseEstimator:
        self.model = AdaBoostClassifier(**self.MODELS[self.name]["best_hyper_params"])
