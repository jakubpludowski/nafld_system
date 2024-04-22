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

THRESHOLD_TO_BE_PART_OF_ENSEMBLE = 0.85

MODELS_NAMES_TO_RENAME = {
    "random_forest_org": "Random Forest",
    "decision_tree_org": "Decision Tree",
    "knn_org": "K-Nearest Neighbors",
    "log_reg_org": "Logistic Regression",
    "svm_org": "Support Vector Machine",
    "adaboost_org": "Ada Boost",
    "mlp_org": "Multi-Layer Perceptron",
    "xgb_org": "XGBoost",
    "random_forest_pca": "Random Forest",
    "decision_tree_pca": "Decision Tree",
    "knn_pca": "K-Nearest Neighbors",
    "log_reg_pca": "Logistic Regression",
    "svm_pca": "Support Vector Machine",
    "adaboost_pca": "Ada Boost",
    "mlp_pca": "Multi-Layer Perceptron",
    "xgb_pca": "XGBoost",
}
