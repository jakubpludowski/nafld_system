import pandas as pd
from nafld.models.data_preparation import prepare_data
from nafld.system.model_funcionality import (
    load_all_models,
    overwrite_models,
    set_model_to_warm_start,
    test_ensemble_model,
    train_all_models,
    validate_all_models,
)
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment
from runscripts.manage_data.configs.step_1_4_config import MODELS_OBJECTS, MODELS_TO_TRAIN

if __name__ == "__main__":
    CONF, MODELS_PARAMS = initialize_environment()

    base_features_table = StaticTable(name="base_features", path_to_table=CONF.DATA_BASE)

    preprocessed_data = prepare_data(base_features_table.read(), perform_shap_analysis=CONF.perform_shap_analysis)

    run_details = []
    all_models = []

    # Create all model objects
    for model_name in MODELS_TO_TRAIN:
        model_name = model_name + "_org" if CONF.perform_shap_analysis else model_name + "_pca"  # noqa: PLW2901
        run_details.append(model_name)
        model = MODELS_OBJECTS[model_name](
            model_name, MODELS_PARAMS, CONF.DATA_MODELS_DIRECTORY, CONF.PATH_TO_BEST_PARAMETERS, CONF.warm_start
        )
        all_models.append(model)

    run_details = pd.DataFrame(run_details, columns=["ModelName"])

    # Load estimators from files
    run_details, all_models = load_all_models(run_details, preprocessed_data, all_models, CONF)
    if CONF.new_data:
        # Set warm start depending on local config
        all_models = set_model_to_warm_start(all_models)

        # Set warm start depending on local config
        all_models = train_all_models(all_models, preprocessed_data)

        # Validate all new trained models, assign new f1 metric.
        run_details, all_models = validate_all_models(run_details, all_models, preprocessed_data)

        # If new f1 is better than previous save new model
        overwrite_models(run_details, all_models)

    test_ensemble_model(all_models, preprocessed_data)
