from nafld.models.abstract_model import AbstractModel
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(AbstractModel):
    def create_new_model(self) -> BaseEstimator:
        self.model = LogisticRegression(**self.MODELS[self.name]["best_hyper_params"])
