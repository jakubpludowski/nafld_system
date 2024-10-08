import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from nafld.models.configs.models_config import PCA_THRESHOLD, RANDOM_STATE, TEST_SIZE
from nafld.table.processed_table import ProcessedPatientFeaturesColumns
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


def prepare_data(
    data: DataFrame, path_for_scaler: str, path_for_pca: str, perform_shap_analysis: bool = False
) -> DataFrame:
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
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    with Path.open(Path(path_for_scaler), "wb") as file:
        pickle.dump(obj=scaler, file=file)

    X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns)
    X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns)
    if not perform_shap_analysis:
        X_train, X_test = perform_PCA(data=(X_train, X_test), path_for_pca=path_for_pca)
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        return (X_train, y_train, X_test, y_test), None

    return (X_train, y_train, X_test, y_test), X.columns


def perform_PCA(data: tuple[DataFrame, DataFrame], path_for_pca: str) -> tuple[DataFrame, DataFrame]:
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
    pca.fit(X_train)
    with Path.open(Path(path_for_pca), "wb") as file:
        pickle.dump(obj=pca, file=file)

    return pca.transform(X_train), pca.transform(X_test)
