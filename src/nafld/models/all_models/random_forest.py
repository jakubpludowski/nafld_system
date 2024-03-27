import copy

from nafld.models.abstract_model import AbstractModel
from nafld.models.configs.models_config import CV_FOR_RANDOM_SEARCH, N_ITER, RANDOM_STATE
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


class RandomForestModel(AbstractModel):
    def __init__(self, name: str, global_models: dict, path_to_models: str, path_to_best_params: str) -> None:
        super().__init__(name, global_models, path_to_models, path_to_best_params)

    def create_new_model(self) -> BaseEstimator:
        self.model = RandomForestClassifier(**self.MODELS[self.name]["best_hyper_params"])

    def get_hyper_parameters(self, data: tuple) -> dict:
        (X_train, y_train, X_test, y_test) = data

        # Check previous model accuracy
        y_pred = self.make_predictions(X_test)
        previous_model_score = accuracy_score(y_test, y_pred)

        # Find new hyper parameters
        param_dist = self.MODELS[self.name]["search_hyper_params"]
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            n_iter=N_ITER,
            cv=CV_FOR_RANDOM_SEARCH,
            random_state=RANDOM_STATE,
        )
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_

        # Check new model accuracy
        model_with_new_params = copy.deepcopy(self.model).set_params(**best_params)
        model_with_new_params.fit(X_train, y_train)
        y_pred = model_with_new_params.predict(X_test)
        new_model_score = accuracy_score(y_test, y_pred)

        # Decide if new hyper parameters should replace old ones
        if new_model_score > previous_model_score:
            self.MODELS[self.name]["best_hyper_params"] = best_params

            self.save_new_best_parameters(self.MODELS)

            return self.MODELS[self.name]["best_hyper_params"]

        return self.MODELS[self.name]["best_hyper_params"]
