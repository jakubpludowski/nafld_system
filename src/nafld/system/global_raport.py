import dalex as dx
from matplotlib import pyplot as plt
from nafld.models.all_models.ensemble import EnsembleModel
from pandas import DataFrame
from sklearn.metrics import auc, roc_curve


def generate_global_raport(model: EnsembleModel, data: DataFrame, results: tuple, models_results: DataFrame) -> None:
    perform_shap_analysis(model=model, data=data)

    models_raw_predictions, ensemble_results, ensemble_auc_results, mean_f1_result = results

    table_with_simple_models_results = models_results.to_html(classes="table table-striped", index=False)

    fpr, tpr, _ = roc_curve(models_raw_predictions["label"], models_raw_predictions["mean_preds"])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="Krzywa ROC (AUC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Krzywa ROC")
    plt.legend(loc="lower right")

    plt.savefig("plots/wykres_roc.png")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raport z wykresami i tabelką</title>
        <style>
            table {{
                border-collapse: collapse;
                width: 60%;
            }}
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
        <h1>Raport z wykresami i tabelką</h1>
        <div>
            <h2>Tabela</h2>
            {table_with_simple_models_results}
        </div>
        <div>
            <h2>Wykres ROC</h2>
            <img src="plots/wykres_roc.png">
        </div>
    </body>
    </html>
    """

    with open("raport.html", "w") as file:  # noqa: PTH123
        file.write(html)


def perform_shap_analysis(model: EnsembleModel, data: DataFrame) -> None:
    (X_train, y_train, X_test, y_test) = data

    exp = dx.Explainer(model.model, X_train, y_train)
    exp.model_performance(model_type="classification")
