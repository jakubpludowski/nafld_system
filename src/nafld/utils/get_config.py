from typing import Any

from .argument_parsing import parse_arguments


class ConfigBase:
    DATA_RAW_DIRECTORY = "data_raw/"
    DATA_PROCESSED_DIRECTORY = "data_processed/"
    DATA_ORIGINAL_DIRECTORY = "data/data_original/"

    ORIGINAL_DATA_CSV_FILE = "data/data_original/data_complete.csv"

    FEATURES_PARQUET = "data.parquet"


class DevConfig(ConfigBase):
    def __init__(self, mode: str) -> None:
        self.MODE = mode

        dir_name = "data/dev/"
        self.DATA_PROCESSED_DIRECTORY = dir_name + self.DATA_PROCESSED_DIRECTORY
        self.DATA_RAW_DIRECTORY = dir_name + self.DATA_RAW_DIRECTORY
        self.RAW_FEATURES_TABLE_PARQUET = self.DATA_RAW_DIRECTORY + self.FEATURES_PARQUET


class ProdConfig(ConfigBase):
    def __init__(self, mode: str) -> None:
        self.MODE = mode

        dir_name = "data/prod/"
        self.DATA_PROCESSED_DIRECTORY = dir_name + self.DATA_PROCESSED_DIRECTORY
        self.DATA_RAW_DIRECTORY = dir_name + self.DATA_RAW_DIRECTORY
        self.RAW_FEATURES_TABLE_PARQUET = dir_name + self.DATA_RAW_DIRECTORY + self.FEATURES_PARQUET


CONFIG_ENVS = {"dev": DevConfig, "prod": ProdConfig}


def get_config(use_args: bool = True, **kwargs: dict[str, Any]) -> ConfigBase:
    if use_args:
        cli_arguments = parse_arguments()
        kwargs.update(vars(cli_arguments))
    mode = kwargs.get("mode")

    return CONFIG_ENVS[mode](**kwargs)
