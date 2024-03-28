from nafld.models.abstract_model import AbstractModel
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModel(AbstractModel):
    def create_new_model(self) -> BaseEstimator:
        self.model = DecisionTreeClassifier(**self.MODELS[self.name]["best_hyper_params"])
