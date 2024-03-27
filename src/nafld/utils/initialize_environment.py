from nafld.config.base_config import ConfigBase, get_config
from nafld.models.configs.models_config import MODELS
from nafld.utils.model_utils.best_parameters_loader import load_best_parameters


def initialize_environment() -> tuple[ConfigBase, dict]:
    try:
        from local_config import config_args
    except ModuleNotFoundError:
        config_args = {}

    CONF = get_config(**config_args)

    path_to_best_parameters = CONF.PATH_TO_BEST_PARAMETERS
    models_best_parameters = MODELS
    GLOBAL_MODELS = load_best_parameters(path_to_best_parameters, models_best_parameters)

    return CONF, GLOBAL_MODELS
