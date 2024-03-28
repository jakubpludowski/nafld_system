import numpy as np
from nafld.models.configs.models_config import PCA_THRESHOLD, RANDOM_STATE, TEST_SIZE
from nafld.table.processed_table import ProcessedPatientFeaturesColumns
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


def prepare_data(data: DataFrame, pca: bool = True) -> DataFrame:
    data = data.drop(columns=(ProcessedPatientFeaturesColumns.PatiendId))
    X = data.drop(columns=[ProcessedPatientFeaturesColumns.Label])
    y = data[ProcessedPatientFeaturesColumns.Label]

    # Split to train and test data with label proportion
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Data standarization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if pca:
        X_train, X_test = perform_PCA(data=(X_train, X_test))

    return (X_train, y_train, X_test, y_test)


def perform_PCA(data: tuple[DataFrame, DataFrame]) -> tuple[DataFrame, DataFrame]:
    X_train, X_test = data
    if not X_train.shape[1] == X_test.shape[1]:
        raise ValueError("Nr of attributes in train and test set does not match")

    for n in range(1, X_train.shape[1]):
        pca = PCA(n_components=n)
        pca.fit(X_train)

        explained_variance = np.sum(pca.explained_variance_ratio_)

        if explained_variance > PCA_THRESHOLD:
            nr_of_components = n
            break

    pca = PCA(n_components=nr_of_components)

    return pca.fit_transform(X_train), pca.transform(X_test)
