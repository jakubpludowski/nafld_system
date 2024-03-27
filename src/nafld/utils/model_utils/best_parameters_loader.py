import json
from pathlib import Path


def save_best_parameters(path_to_file: str, model_dict: dict) -> None:
    best_params = {}

    for model_name, model_data in model_dict.items():
        best_hyper_params = model_data.get("best_hyper_params")
        best_params.update({model_name: best_hyper_params})

    with Path.open(Path(path_to_file), "w") as f:
        json.dump(best_params, f)


def load_best_parameters(path_to_file: str, model_dict: dict) -> dict:
    with Path.open(Path(path_to_file), "r") as f:
        best_params = json.load(f)

    for model_name, model_data in best_params.items():
        if model_data:
            model_dict[model_name]["best_hyper_params"] = model_data

    return model_dict
