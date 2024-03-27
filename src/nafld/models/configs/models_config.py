import xgboost as xgb
from scipy.stats import randint
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

MODELS = {
    "random_forest": {
        "obj": RandomForestClassifier(),
        "best_hyper_params": {},
        "search_hyper_params": {
            "n_estimators": randint(10, 100),
            "max_depth": randint(1, 10),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["auto", "sqrt", "log2"],
            "criterion": ["gini", "entropy"],
        },
    },
    "knn": {"obj": KNeighborsClassifier()},
    "log_reg": {"obj": LogisticRegression()},
    "svm": {"obj": SVC()},
    "adaboost": {"obj": AdaBoostClassifier()},
    "mlp": {"obj": MLPClassifier()},
    "xgb": {"obj": xgb.Booster()},
}

RANDOM_STATE = 300464
TEST_SIZE = 0.2
PCA_THRESHOLD = 0.9
N_ITER = 100
CV_FOR_RANDOM_SEARCH = 5
