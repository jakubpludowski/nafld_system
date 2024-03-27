import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
    "decision_tree": {
        "obj": DecisionTreeClassifier(),
        "best_hyper_params": {},
        "search_hyper_params": {
            "max_depth": [3, 4, 5, 6, 7, 8, None],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["auto", "sqrt", "log2"],
            "criterion": ["gini", "entropy"],
        },
    },
    "knn": {
        "obj": KNeighborsClassifier(),
        "best_hyper_params": {},
        "search_hyper_params": {
            "n_neighbors": randint(1, 20),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": randint(10, 50),
            "p": [1, 2],
        },
    },
    "log_reg": {
        "obj": LogisticRegression(),
        "best_hyper_params": {},
        "search_hyper_params": {
            "C": loguniform(0.001, 100),
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
            "max_iter": [100, 200, 300],
        },
    },
    "svm": {
        "obj": SVC(),
        "best_hyper_params": {},
        "search_hyper_params": {
            "C": loguniform(0.001, 100),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": loguniform(0.0001, 1),
            "degree": [2, 3, 4],
            "coef0": [0.0, 1.0, 2.0],
        },
    },
    "adaboost": {
        "obj": AdaBoostClassifier(),
        "best_hyper_params": {},
        "search_hyper_params": {
            "n_estimators": randint(50, 500),
            "learning_rate": uniform(0.01, 1.0),
            "algorithm": ["SAMME"],
        },
    },
    "mlp": {
        "obj": MLPClassifier(),
        "best_hyper_params": {},
        "search_hyper_params": {
            "hidden_layer_sizes": [(100,), (50, 50), (100, 50, 25)],
            "activation": ["logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": uniform(0.0001, 0.1),
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": uniform(0.001, 0.1),
            "max_iter": randint(100, 1000),
        },
    },
    "xgb": {
        "obj": xgb.Booster(),
        "best_hyper_params": {},
        "search_hyper_params": {
            "n_neighbors": randint(1, 20),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": randint(10, 50),
            "p": [1, 2],
        },
    },
}

RANDOM_STATE = 300464
TEST_SIZE = 0.2
PCA_THRESHOLD = 0.9
N_ITER = 100
CV_FOR_RANDOM_SEARCH = 5
