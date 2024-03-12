from typing import Any

from .argument_parsing import parse_arguments


class ConfigBase:
    DATA_RAW_DIRECTORY = "raw/"
    DATA_BASE_DIRECTORY = "base/"
    DATA_MODELS_DIRECTORY = "models/"
    DATA_INPUTS_DIRECTORY = "inputs/"

    INPUTS_NEW_DIRECTORY = "new/"
    INPUTS_ORIGINAL_DIRECTORY = "original/"
    INPUTS_PATIENTS_DIRECTORY = "patients/"
    INPUTS_PROCESSED = "processed/"

    ORIGINAL_DATA_CSV_FILE = "data/data_original/data_complete.csv"

    FEATURES_PARQUET = "data.parquet"
    FEATURES_CSV = "data.csv"
    INPUT_DATA_CSV = "data.csv"
    INPUT_ORIGINAL_DATA_CSV = "data.csv"

    def __init__(self, mode: str) -> None:
        data_dir_name = f"data/{mode}/"
        self.DATA_BASE_DIRECTORY = data_dir_name + self.DATA_BASE_DIRECTORY
        self.DATA_RAW_DIRECTORY = data_dir_name + self.DATA_RAW_DIRECTORY
        self.DATA_MODELS_DIRECTORY = data_dir_name + self.DATA_MODELS_DIRECTORY
        self.DATA_INPUTS_DIRECTORY = data_dir_name + self.DATA_INPUTS_DIRECTORY

        self.DATA_INPUTS_NEW_DIRECTORY = self.DATA_INPUTS_DIRECTORY + self.INPUTS_NEW_DIRECTORY
        self.DATA_INPUTS_ORIGINAL_DIRECTORY = self.DATA_INPUTS_DIRECTORY + self.INPUTS_ORIGINAL_DIRECTORY
        self.DATA_INPUTS_PATIENTS_DIRECTORY = self.DATA_INPUTS_DIRECTORY + self.INPUTS_PATIENTS_DIRECTORY
        self.DATA_INPUTS_PROCESSED_DIRECTORY = self.DATA_INPUTS_DIRECTORY + self.INPUTS_PROCESSED

        self.DATA_BASE = self.DATA_BASE_DIRECTORY + self.FEATURES_PARQUET
        self.DATA_RAW = self.DATA_RAW_DIRECTORY + self.FEATURES_CSV

        # TODO: add models

        self.INPUTS_ORIGINAL_DATA = self.DATA_INPUTS_ORIGINAL_DIRECTORY + self.INPUT_ORIGINAL_DATA_CSV
        self.INPUTS_NEW_DATA = self.DATA_INPUTS_NEW_DIRECTORY + self.INPUT_DATA_CSV
        self.INPUTS_PROCESSED_DATA = self.DATA_INPUTS_PROCESSED_DIRECTORY + self.INPUT_DATA_CSV


class DevConfig(ConfigBase):
    def __init__(self, mode: str) -> None:
        self.MODE = mode
        super().__init__(mode)


class TestConfig(ConfigBase):
    def __init__(self, mode: str) -> None:
        self.MODE = mode
        super().__init__(mode)


class ProdConfig(ConfigBase):
    def __init__(self, mode: str) -> None:
        self.MODE = mode
        super().__init__(mode)


CONFIG_ENVS = {"dev": DevConfig, "prod": ProdConfig}


def get_config(use_args: bool = True, **kwargs: dict[str, Any]) -> ConfigBase:
    if use_args:
        cli_arguments = parse_arguments()
        kwargs.update(vars(cli_arguments))
    mode = kwargs.get("mode")

    return CONFIG_ENVS[mode](**kwargs)
