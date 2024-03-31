from nafld.models.all_models import (
    AdaBoostModel,
    DecisionTreeModel,
    KNNModel,
    LogisticRegressionModel,
    MLPModel,
    RandomForestModel,
    SupportVectorMachineModel,
    XGBoostModel,
)

MODELS_TO_TRAIN = ["adaboost", "decision_tree", "knn"]
# MODELS_TO_TRAIN = ["random_forest", "decision_tree",
# "knn", "log_reg", "svm", "adaboost", "mlp", "xgb"]


MODELS_OBJECTS = {
    "random_forest": RandomForestModel,
    "decision_tree": DecisionTreeModel,
    "knn": KNNModel,
    "log_reg": LogisticRegressionModel,
    "svm": SupportVectorMachineModel,
    "adaboost": AdaBoostModel,
    "mlp": MLPModel,
    "xgb": XGBoostModel,
}
