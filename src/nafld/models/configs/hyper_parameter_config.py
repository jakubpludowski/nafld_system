from scipy.stats import loguniform, randint, uniform

MODELS = {
    "random_forest_org": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "n_estimators": randint(10, 100),
            "max_depth": randint(1, 10),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2"],
            "criterion": ["gini", "entropy"],
        },
    },
    "decision_tree_org": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "max_depth": [3, 4, 5, 6, 7, 8, None],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2"],
            "criterion": ["gini", "entropy"],
        },
    },
    "knn_org": {
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
    "log_reg_org": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "C": loguniform(0.001, 100),
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
            "max_iter": [100, 200, 300],
        },
    },
    "svm_org": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "C": loguniform(0.001, 100),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": loguniform(0.0001, 1),
            "degree": [2, 3, 4],
            "coef0": [0.0, 1.0, 2.0],
        },
    },
    "adaboost_org": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "n_estimators": randint(50, 500),
            "learning_rate": uniform(0.01, 1.0),
            "algorithm": ["SAMME"],
        },
    },
    "mlp_org": {
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
    "xgb_org": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "learning_rate": uniform(0.01, 0.3),
            "max_depth": randint(3, 10),
            "n_estimators": randint(100, 1000),
            "min_child_weight": [1, 5, 10],
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "gamma": [0, 0.5, 1],
            "reg_alpha": [0, 0.001, 0.01, 0.1, 1, 10],
            "reg_lambda": [0, 0.001, 0.01, 0.1, 1, 10],
        },
    },
    "random_forest_pca": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "n_estimators": randint(10, 100),
            "max_depth": randint(1, 10),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2"],
            "criterion": ["gini", "entropy"],
        },
    },
    "decision_tree_pca": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "max_depth": [3, 4, 5, 6, 7, 8, None],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2"],
            "criterion": ["gini", "entropy"],
        },
    },
    "knn_pca": {
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
    "log_reg_pca": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "C": loguniform(0.001, 100),
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
            "max_iter": [100, 200, 300],
        },
    },
    "svm_pca": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "C": loguniform(0.001, 100),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": loguniform(0.0001, 1),
            "degree": [2, 3, 4],
            "coef0": [0.0, 1.0, 2.0],
        },
    },
    "adaboost_pca": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "n_estimators": randint(50, 500),
            "learning_rate": uniform(0.01, 1.0),
            "algorithm": ["SAMME"],
        },
    },
    "mlp_pca": {
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
    "xgb_pca": {
        "best_hyper_params": {},
        "search_hyper_params": {
            "learning_rate": uniform(0.01, 0.3),
            "max_depth": randint(3, 10),
            "n_estimators": randint(100, 1000),
            "min_child_weight": [1, 5, 10],
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "gamma": [0, 0.5, 1],
            "reg_alpha": [0, 0.001, 0.01, 0.1, 1, 10],
            "reg_lambda": [0, 0.001, 0.01, 0.1, 1, 10],
        },
    },
}
