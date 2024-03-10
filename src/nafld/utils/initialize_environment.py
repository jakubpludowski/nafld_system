from nafld.utils.get_config import ConfigBase, get_config


def initialize_environment() -> tuple[ConfigBase]:
    try:
        from local_config import config_args
    except ModuleNotFoundError:
        config_args = {}

    return get_config(**config_args)
