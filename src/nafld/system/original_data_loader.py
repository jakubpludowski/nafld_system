import numpy as np
from nafld.table.tables.static_table import StaticTable
from pandas import DataFrame, get_dummies
from runscripts.manage_data.configs.step_1_2_config import (
    AGE_UNTRUSTY,
    COLNAMES_DICT_TO_TRANSLATE,
    COLUMN_NAMES_TO_DROP,
    COLUMNS_TO_FILL_WITH_VALUES_FROM_NORMAL_DISTRIBUTION,
    RANDOM_STATE_FOR_REGRESSION_MODELS,
    TEST_SIZE_FOR_REGRESSION_MODELS,
    WEIGHT_UNTRUSTY,
)
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    def __init__(self, input_table: StaticTable, output_table: StaticTable, mode: str, seed: int) -> None:
        self.input_table = input_table
        self.output_table = output_table
        self.mode = mode
        if seed:
            np.random.seed(seed)

    def save_table(self, data: DataFrame) -> None:
        self.output_table.write_csv(data)

    def load_data_from_table(self) -> DataFrame:
        return self.input_table.read(file_format="excel")

    def process_data(self) -> None:
        df = self.load_data_from_table()
        df = self.delete_empty_rows_and_columns(df)
        df = self.drop_redundant_columns(df)
        df = self.manage_label(df)
        df = self.handle_sex_column(df)
        df = self.replace_untrusty_records(df)
        df = self.fill_feature_with_value_from_distribution(df, COLUMNS_TO_FILL_WITH_VALUES_FROM_NORMAL_DISTRIBUTION)

        df = self.fill_feature_based_on_regression_model(df, "weight", ["age"])
        df = self.fill_feature_based_on_regression_model(df, "height", ["age", "weight"])
        df = self.handle_BMI(df)

        df = self.deal_with_categorical_columns(df)

        df = df.rename(columns=COLNAMES_DICT_TO_TRANSLATE)

        self.save_table(df)

    def delete_empty_rows_and_columns(self, df: DataFrame) -> DataFrame:
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        return df.reset_index().drop(columns=["index"])

    def drop_redundant_columns(self, df: DataFrame) -> DataFrame:
        return df.drop(columns=COLUMN_NAMES_TO_DROP)

    def manage_label(self, df: DataFrame) -> DataFrame:
        df.loc[df["label"] == " s", "label"] = "s"
        df.loc[df["label"] == "S", "label"] = "s"
        if self.mode == "binary":
            df.loc[df["label"] == "o", "label"] = "k"
            df["label"] = df["label"].apply(binarize_label)
        else:
            df["label"] = df["label"].apply(binarize_label, binary=False)

        return df

    def replace_untrusty_records(self, df: DataFrame) -> DataFrame:
        trusty_data = df[~((df["age"] < AGE_UNTRUSTY) & (df["weight"] > WEIGHT_UNTRUSTY))]
        mean = trusty_data["age"].mean()
        std = trusty_data["age"].std()
        nan_indices = (df["age"] < AGE_UNTRUSTY) & (df["weight"] > WEIGHT_UNTRUSTY)
        new_values = np.random.normal(loc=mean, scale=std, size=len(df))
        df.loc[nan_indices, "age"] = new_values[nan_indices]
        df.loc[nan_indices, "weight"] = np.nan

        return df

    def fill_feature_with_value_from_distribution(self, df: DataFrame, columns: str) -> DataFrame:
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            nan_indices = df[column].isna()
            new_values = np.random.normal(loc=mean, scale=std, size=len(df))
            df.loc[nan_indices, column] = new_values[nan_indices]
        return df

    def fill_feature_based_on_regression_model(
        self, dataframe_to_fill: DataFrame, column_to_fill: str, column_to_base_model_on: list[str]
    ) -> DataFrame:
        df = dataframe_to_fill.copy()
        X = df.dropna(subset=[column_to_fill])[column_to_base_model_on]
        y = df.dropna(subset=[column_to_fill])[column_to_fill]
        X_predict = df[df[column_to_fill].isnull()][column_to_base_model_on]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE_FOR_REGRESSION_MODELS, random_state=RANDOM_STATE_FOR_REGRESSION_MODELS
        )

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_predict = scaler.transform(X_predict)

        alphas = np.arange(0.001, 10, 0.01)
        if len(column_to_base_model_on) == 1:
            model = RidgeCV(alphas=alphas, scoring="neg_root_mean_squared_error", cv=4, fit_intercept=True)
        else:
            model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], tol=0.01)
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)

        var = root_mean_squared_error(y_train, train_predictions)

        test_predictions = model.predict(X_test)

        MAE = mean_absolute_error(y_test, test_predictions)
        RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))

        print(f"MAE: {MAE}\nRMSE: {RMSE}")  # noqa: T201

        predicted_values = model.predict(X_predict)

        random_values = np.random.normal(loc=predicted_values, scale=var)

        df.loc[df[column_to_fill].isnull(), column_to_fill] = random_values

        return df

    def handle_BMI(self, df: DataFrame) -> DataFrame:
        df.loc[df["BMI"].isnull(), ["BMI"]] = df[df["BMI"].isnull()]["weight"] / (df[df["BMI"].isnull()]["height"] ** 2)
        return df

    def handle_sex_column(self, df: DataFrame) -> DataFrame:
        df.loc[df["sex"] != "D", ["sex"]] = "C"
        df["sex"] = get_dummies(df["sex"], drop_first=True)

        return df

    def deal_with_categorical_columns(self, df: DataFrame) -> DataFrame:
        df.loc[df["bilirubina"] == "<1,0", ["bilirubina"]] = 0.5
        df.loc[df["bilirubina"] == "< 1,0", ["bilirubina"]] = 0.5
        df.loc[df["bilirubina"].isnull(), ["bilirubina"]] = 1
        df["bilirubina"] = df["bilirubina"].apply(float)

        df.loc[df["bil_bezpośrednia"] == "<1,0", ["bil_bezpośrednia"]] = 0.5
        df.loc[df["bil_bezpośrednia"] == "< 1,0", ["bil_bezpośrednia"]] = 0.5
        df.loc[df["bil_bezpośrednia"].isnull(), ["bil_bezpośrednia"]] = 1
        df["bil_bezpośrednia"] = df["bil_bezpośrednia"].apply(float)

        return df


def binarize_label(old_label: str, binary: bool = True) -> None:
    default_label = 999
    new_lable_for_o = 1 if binary else 2
    labels = {
        "s": 0,
        "k": 1,
        "o": new_lable_for_o,
    }
    return labels.get(old_label, default_label)
