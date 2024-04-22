import pickle
from pathlib import Path

import dalex as dx
import pandas as pd
from nafld.models.all_models.ensemble import EnsembleModel
from nafld.system.diagnosis import check_if_patient_is_healthy
from nafld.table.processed_table import ProcessedPatientFeaturesColumns
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment

if __name__ == "__main__":
    CONF, MODELS_PARAMS = initialize_environment()

    patients_table = StaticTable(name="patients", path_to_table=CONF.PATIENTS_TO_DIAGNOSE_BASE)
    patients = patients_table.read()
    label = patients[ProcessedPatientFeaturesColumns.Label]
    ids = patients[ProcessedPatientFeaturesColumns.PatiendId].to_list()
    data = patients.drop(columns=(ProcessedPatientFeaturesColumns.PatiendId))
    X = data.drop(columns=[ProcessedPatientFeaturesColumns.Label])

    with Path.open(Path(CONF.PATH_TO_MODEL_SCALER), "rb") as file:
        scaler = pickle.load(file)  # noqa: S301
    scaled_X = scaler.transform(X)
    patients = pd.DataFrame(scaled_X, columns=X.columns)

    ensemble_model = EnsembleModel(
        "ensemble", None, CONF.PATH_TO_ALL_MODELS, CONF.PATH_TO_BEST_PARAMETERS, CONF.warm_start
    )

    _, check, _ = ensemble_model.load_model()
    if check:
        raise ValueError("Ensemble model has not been trained yet")
    with Path.open(Path(CONF.PATH_TO_MODEL_EXPLAINER), "rb") as file:
        model_explainer = dx.Explainer.load(file)

    for index, row in patients.iterrows():
        single_patient = pd.DataFrame(row).transpose()
        path = CONF.DATA_DIAGNOSIS_DIRECTORY + f"pat{ids[index]}.html"

        output_from_exp = model_explainer.predict(single_patient)[0]
        output_from_model = ensemble_model.model.predict(single_patient)[0]
        certainty, diagnosis = check_if_patient_is_healthy(output_from_exp)

        if CONF.wide_diagnosis:
            bd = model_explainer.predict_parts(single_patient, type="break_down", label=f"Patient nr {ids[index]}")
            bd_interactions = model_explainer.predict_parts(
                single_patient, type="break_down_interactions", label=f"Patient nr {ids[index]}"
            )
            sh = model_explainer.predict_parts(single_patient, type="shap", B=10, label=f"Patient nr {ids[index]}")

            fig = bd.plot(bd_interactions, show=False)
            path1 = f"images/pat{ids[index]}_1.png"
            fig.write_image(CONF.DATA_DIAGNOSIS_DIRECTORY + path1)

            fig = sh.plot(bar_width=16, show=False)
            path2 = f"images/pat{ids[index]}_2.png"
            fig.write_image(CONF.DATA_DIAGNOSIS_DIRECTORY + path2)

            html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Diagnoza dla pacjenta {ids[index]}</title>
                        <style>
                            th, td {{
                                border: 1px solid black;
                                padding: 8px;
                                text-align: left;
                            }}
                            th {{
                                background-color: #f2f2f2;
                            }}
                        </style>
                    </head>
                    <body>
                        <h1>Diagnoza pacjenta {ids[index]}</h1>
                        <div>
                            <h2>Pacjent jest {diagnosis}</h2>
                            <h2>Model zdiagnozował pacjenta z pewnością {certainty}</h2>
                        </div>
                        <div>
                            <h3>Modele określiły stan pacjenta na podstawie atrybutów:</h3>
                            <h4>Wartości Shapleya. Jak wartości atrybutów pwłynęły na ostateczny wynik diagnozy:</h4>
                            <img src="{path2}", alt = "Shapley values">
                        </div>
                        <div>
                            <h4>Jak model myślał:</h4>
                            <img src="{path1}", alt = "Model break down">
                        </div>
                    </body>
                    </html>
                    """
        else:
            html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Diagnoza dla pacjenta {ids[index]}</title>
                        <style>
                            th, td {{
                                border: 1px solid black;
                                padding: 8px;
                                text-align: left;
                            }}
                            th {{
                                background-color: #f2f2f2;
                            }}
                        </style>
                    </head>
                    <body>
                        <h1>Diagnoza pacjenta {ids[index]}</h1>
                        <div>
                            <h2>Pacjent jest {diagnosis}</h2>
                            <h2>Model zdiagnozował pacjenta z pewnością {certainty}</h2>
                        </div>
                    </body>
                    </html>
                    """

        with open(path, "w") as file:  # noqa: PTH123
            file.write(html)
