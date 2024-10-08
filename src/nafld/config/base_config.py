from typing import Any

from nafld.utils.argument_parsing import parse_arguments


class ConfigBase:
    DATA_RAW_DIRECTORY = "raw/"
    DATA_BASE_DIRECTORY = "base/"
    DATA_MODELS_DIRECTORY = "models/"
    DATA_INPUTS_DIRECTORY = "inputs/"
    DATA_DIAGNOSIS = "diagnosis/"

    INPUTS_NEW_DIRECTORY = "new/"
    INPUTS_ORIGINAL_DIRECTORY = "original/"
    INPUTS_PATIENTS_DIRECTORY = "patients/"
    INPUTS_PROCESSED = "processed/"

    ORIGINAL_DATA_CSV_FILE = "data/data_original/data_complete.csv"

    FEATURES_PARQUET = "data.parquet"
    NEW_FEATURES_PARQUET = "new_data.parquet"
    FEATURES_CSV = "data.csv"
    INPUT_DATA_CSV = "data.csv"
    INPUT_ORIGINAL_DATA_XLSX = "data_original.xlsx"

    def __init__(
        self,
        mode: str,
        warm_start: bool = True,
        tune_hyperparams: bool = True,
        wide_diagnosis: bool = False,
        perform_shap_analysis: bool = False,
    ) -> None:
        # Paths
        data_dir_name = f"data/{mode}/"
        self.DATA_BASE_DIRECTORY = data_dir_name + self.DATA_BASE_DIRECTORY
        self.DATA_RAW_DIRECTORY = data_dir_name + self.DATA_RAW_DIRECTORY
        self.DATA_MODELS_DIRECTORY = data_dir_name + self.DATA_MODELS_DIRECTORY
        self.DATA_INPUTS_DIRECTORY = data_dir_name + self.DATA_INPUTS_DIRECTORY
        self.DATA_DIAGNOSIS_DIRECTORY = data_dir_name + self.DATA_DIAGNOSIS

        self.DATA_INPUTS_NEW_DIRECTORY = self.DATA_INPUTS_DIRECTORY + self.INPUTS_NEW_DIRECTORY
        self.DATA_INPUTS_ORIGINAL_DIRECTORY = self.DATA_INPUTS_DIRECTORY + self.INPUTS_ORIGINAL_DIRECTORY
        self.DATA_INPUTS_PATIENTS_DIRECTORY = self.DATA_INPUTS_DIRECTORY + self.INPUTS_PATIENTS_DIRECTORY
        self.DATA_INPUTS_PROCESSED_DIRECTORY = self.DATA_INPUTS_DIRECTORY + self.INPUTS_PROCESSED

        self.DATA_BASE = self.DATA_BASE_DIRECTORY + self.FEATURES_PARQUET
        self.NEW_DATA_BASE = self.DATA_BASE_DIRECTORY + self.NEW_FEATURES_PARQUET
        self.DATA_RAW = self.DATA_RAW_DIRECTORY + self.FEATURES_CSV

        self.INPUTS_ORIGINAL_DATA = self.DATA_INPUTS_ORIGINAL_DIRECTORY + self.INPUT_ORIGINAL_DATA_XLSX
        self.INPUTS_NEW_DATA = self.DATA_INPUTS_NEW_DIRECTORY + self.INPUT_DATA_CSV
        self.INPUTS_PROCESSED_DATA = self.DATA_INPUTS_PROCESSED_DIRECTORY + self.INPUT_DATA_CSV

        self.PATH_TO_BEST_PARAMETERS = self.DATA_MODELS_DIRECTORY + "best_params.json"
        self.PATH_TO_MODEL_EXPLAINER = self.DATA_MODELS_DIRECTORY + "exp.pkl"
        self.PATH_TO_MODEL_PCA = self.DATA_MODELS_DIRECTORY + "pca.pkl"
        self.PATH_TO_MODEL_SCALER = self.DATA_MODELS_DIRECTORY + "scaler.pkl"

        self.PATH_TO_ALL_MODELS = self.DATA_MODELS_DIRECTORY + "all_models/"

        self.PATIENTS_TO_DIAGNOSE_BASE = self.DATA_INPUTS_PATIENTS_DIRECTORY + "patients.parquet"

        # Params from local config
        self.warm_start = warm_start
        self.tune_hyperparams = tune_hyperparams
        self.wide_diagnosis = wide_diagnosis
        self.perform_shap_analysis = perform_shap_analysis


class DevConfig(ConfigBase):
    def __init__(self, mode: str, **kwargs: dict[str, Any]) -> None:
        self.MODE = mode
        super().__init__(mode, **kwargs)


class TestConfig(ConfigBase):
    def __init__(self, mode: str, **kwargs: dict[str, Any]) -> None:
        self.MODE = mode
        super().__init__(mode, **kwargs)


class ProdConfig(ConfigBase):
    def __init__(self, mode: str, **kwargs: dict[str, Any]) -> None:
        self.MODE = mode
        super().__init__(mode, **kwargs)


CONFIG_ENVS = {"dev": DevConfig, "prod": ProdConfig}


def get_config(use_args: bool = True, **kwargs: dict[str, Any]) -> ConfigBase:
    if use_args:
        cli_arguments = parse_arguments()
        kwargs.update(vars(cli_arguments))
    mode = kwargs.get("mode")

    return CONFIG_ENVS[mode](**kwargs)
