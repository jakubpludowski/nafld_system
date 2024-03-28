import xgboost as xgb
from nafld.models.abstract_model import AbstractModel
from sklearn.base import BaseEstimator


class XGBoostModel(AbstractModel):
    def create_new_model(self) -> BaseEstimator:
        self.model = xgb.XGBClassifier(**self.MODELS[self.name]["best_hyper_params"])
