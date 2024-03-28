from nafld.models.abstract_model import AbstractModel
from sklearn.base import BaseEstimator
from sklearn.svm import SVC


class SupportVectorMachineModel(AbstractModel):
    def create_new_model(self) -> BaseEstimator:
        self.model = SVC(**self.MODELS[self.name]["best_hyper_params"])
