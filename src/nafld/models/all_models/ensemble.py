from nafld.models.abstract_model import AbstractModel
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier


class EnsembleModel(AbstractModel):
    def create_new_model(self, models: list[tuple[str, BaseEstimator]]) -> BaseEstimator:
        self.model = VotingClassifier(estimators=models, voting="hard")
