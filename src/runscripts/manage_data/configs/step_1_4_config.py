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

# MODELS_TO_TRAIN = ["adaboost", "decision_tree", "knn"]  # noqa: ERA001
MODELS_TO_TRAIN = ["random_forest", "decision_tree", "knn", "log_reg", "svm", "adaboost", "mlp", "xgb"]


MODELS_OBJECTS = {
    "random_forest_org": RandomForestModel,
    "decision_tree_org": DecisionTreeModel,
    "knn_org": KNNModel,
    "log_reg_org": LogisticRegressionModel,
    "svm_org": SupportVectorMachineModel,
    "adaboost_org": AdaBoostModel,
    "mlp_org": MLPModel,
    "xgb_org": XGBoostModel,
    "random_forest_pca": RandomForestModel,
    "decision_tree_pca": DecisionTreeModel,
    "knn_pca": KNNModel,
    "log_reg_pca": LogisticRegressionModel,
    "svm_pca": SupportVectorMachineModel,
    "adaboost_pca": AdaBoostModel,
    "mlp_pca": MLPModel,
    "xgb_pca": XGBoostModel,
}
