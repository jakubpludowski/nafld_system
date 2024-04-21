import pandas as pd
from nafld.models.all_models.ensemble import EnsembleModel
from nafld.models.data_preparation import prepare_data
from nafld.system.global_raport import generate_global_raport
from nafld.system.model_funcionality import (
    load_all_models,
    overwrite_models,
    set_model_to_warm_start,
    test_ensemble_model,
    tidy_run_details,
    train_all_models,
    validate_all_models,
)
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment
from runscripts.manage_data.configs.step_1_4_config import MODELS_OBJECTS, MODELS_TO_TRAIN

if __name__ == "__main__":
    CONF, MODELS_PARAMS = initialize_environment()

    base_features_table = StaticTable(name="base_features", path_to_table=CONF.DATA_BASE)
    new_base_features_table = StaticTable(name="new_base_features", path_to_table=CONF.NEW_DATA_BASE)

    # Decide if we use all data or only data increment
    increment_data = new_base_features_table.read()
    new_data = False
    if CONF.warm_start:
        if not increment_data.empty:
            new_data = True
            data_to_operate = increment_data
        else:
            raise ValueError(
                "There is no new data to perform warm start on.\
                If you want to perform training anyway, set warm start to false."
            )
    else:
        data_to_operate = base_features_table.read()

    # Preprocessing of data
    preprocessed_data, feature_names = prepare_data(data_to_operate, perform_shap_analysis=CONF.perform_shap_analysis)

    # Create all model objects
    run_details = []
    all_models = []
    # Create ensemble model object
    ensemble_model = EnsembleModel(
        "ensemble", None, CONF.DATA_MODELS_DIRECTORY, CONF.PATH_TO_BEST_PARAMETERS, CONF.warm_start
    )

    # Create all single model objects
    for model_name in MODELS_TO_TRAIN:
        model_name = model_name + "_org" if CONF.perform_shap_analysis else model_name + "_pca"  # noqa: PLW2901
        run_details.append(model_name)
        model = MODELS_OBJECTS[model_name](
            model_name, MODELS_PARAMS, CONF.DATA_MODELS_DIRECTORY, CONF.PATH_TO_BEST_PARAMETERS, CONF.warm_start
        )
        all_models.append(model)

    run_details = pd.DataFrame(run_details, columns=["ModelName"])

    # Load estimators from files (or create new ones)
    run_details, all_models = load_all_models(run_details, preprocessed_data, all_models, CONF)

    # Set warm start depending on local config
    all_models = set_model_to_warm_start(all_models)

    # Train all models
    all_models = train_all_models(all_models, preprocessed_data, feature_names)

    # Validate all new trained models, assign new f1 metric.
    run_details, all_models = validate_all_models(run_details, all_models, preprocessed_data)

    # If new f1 is better than previous save new model
    overwrite_models(run_details, all_models)

    all_results = test_ensemble_model(run_details, ensemble_model, all_models, preprocessed_data)
    models_raw_predictions, ensemble_results, ensemble_auc_results, mean_f1_result = all_results

    run_details = tidy_run_details(run_details)

    generate_global_raport(ensemble_model, preprocessed_data, all_results, run_details)
