import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

MODELS = {
    "random_forest": {"obj": RandomForestClassifier()},
    "knn": {"obj": KNeighborsClassifier()},
    "log_reg": {"obj": LogisticRegression()},
    "svm": {"obj": SVC()},
    "adaboost": {"obj": AdaBoostClassifier()},
    "mlp": {"obj": MLPClassifier()},
    "xgb": {"obj": xgb.Booster()},
}
