from pathlib import Path

import dalex as dx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def generate_global_raport(results: tuple, models_results: DataFrame, path_to_explainer: str) -> None:
    (
        models_raw_predictions,
        ensemble_results,
        ensemble_auc_results,
        mean_f1_result,
        feature_importances_per_model,
    ) = results
    confusion_matrix = ensemble_auc_results["confusion_matrix"]

    prepare_confusion_matrix(confusion_matrix)
    generate_roc_plot_for_main_model(models_raw_predictions)

    table_with_simple_models_results = models_results.to_html(classes="table table-striped", index=False)
    main_model_stats_table = prepare_main_model_stats(ensemble_results, ensemble_auc_results).to_html(
        classes="table table-striped", index=False
    )

    random_forest_org_table = feature_importances_per_model["random_forest_org"].to_html(
        classes="table table-striped", index=False
    )
    decision_tree_org_table = feature_importances_per_model["decision_tree_org"].to_html(
        classes="table table-striped", index=False
    )
    adaboost_org_table = feature_importances_per_model["adaboost_org"].to_html(
        classes="table table-striped", index=False
    )
    xgb_org_table = feature_importances_per_model["xgb_org"].to_html(classes="table table-striped", index=False)

    perform_shap_analysis(path_to_explainer=path_to_explainer)

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
        <h1>Raport jakości systemu do wspomagania diagnozy NAFLD</h1>
        <div>
            <h3>Wyniki głównego modelu</h3>
            {main_model_stats_table}
        </div>
        <div>
            <h3>Wykres ROC</h3>
            <img src="plots/wykres_roc_auc.png">
            <img src="plots/wykres_roc_pr.png">
        </div>
        <div>
            <h3>Macierz błędu</h3>
            <img src="plots/macierz_bledu.png">
        </div>
        <h2>Raport z wykresami i tabelką</h2>
        <div>
            <h3>Tabela</h3>
            {table_with_simple_models_results}
        </div>
        <h1>Wyjaśnialność modeli</h1>
        <div>
            <h2>Na jakie parametry główny model zwraca uwagę?</h2>
            <img src="plots/main_model_exp.png">
            <h2>Na jakie parametry zwracają uwagę modele składowe?</h2>
            <h3>Random Forest</h3>
            <img src="plots/plots_per_model/random_forest_org_exp.png">
            {random_forest_org_table}
            <h3>Decision Tree</h3>
            <img src="plots/plots_per_model/decision_tree_org_exp.png">
            {decision_tree_org_table}
            <h3>K-Nearest Neighbors</h3>
            <img src="plots/plots_per_model/knn_org_exp.png">
            <h3>Logistic Regression</h3>
            <img src="plots/plots_per_model/log_reg_org_exp.png">
            <h3>Support Vector Machine</h3>
            <img src="plots/plots_per_model/svm_org_exp.png">
            <h3>Ada Boost</h3>
            <img src="plots/plots_per_model/adaboost_org_exp.png">
            {adaboost_org_table}
            <h3>Multi-Layer Perceptron</h3>
            <img src="plots/plots_per_model/mlp_org_exp.png">
            <h3>XGBoost</h3>
            <img src="plots/plots_per_model/xgb_org_exp.png">
            {xgb_org_table}

        </div>
    </body>
    </html>
    """

    with open("raport.html", "w") as file:  # noqa: PTH123
        file.write(html)


def perform_shap_analysis(path_to_explainer: str) -> None:
    with Path.open(Path(path_to_explainer), "rb") as file:
        exp = dx.Explainer.load(file)
    exp.model_performance(model_type="classification")
    vi = exp.model_parts()
    fig = vi.plot(max_vars=5, show=False)
    fig.write_image("plots/main_model_exp.png")


def prepare_confusion_matrix(cm: np.array) -> None:
    cm_percent = (cm / np.sum(cm)) * 100

    plt.figure()
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predykcje modelu")
    plt.ylabel("Prawdziwe etykiety")
    plt.title("Macierz błędu (w %)")
    plt.savefig("plots/macierz_bledu.png")


def prepare_main_model_stats(ensemble_results: dict, ensemble_auc_results: dict) -> DataFrame:
    ensemble_results = {**ensemble_results, **ensemble_auc_results}
    del ensemble_results["confusion_matrix"]

    main_model_stats = pd.DataFrame([ensemble_results])
    main_model_stats.columns = ["F1", "Accuracy", "Precision", "Recall", "Roc AUC", "AUC PR"]

    return main_model_stats


def generate_roc_plot_for_main_model(models_raw_predictions: DataFrame) -> None:
    fpr, tpr, _ = roc_curve(models_raw_predictions["label"], models_raw_predictions["mean_preds"])
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(models_raw_predictions["label"], models_raw_predictions["mean_preds"])
    auc_pr = auc(recall, precision)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=4, label="Krzywa ROC (AUC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Krzywa ROC")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig("plots/wykres_roc_auc.png")

    plt.figure()
    plt.plot(recall, precision, color="darkorange", lw=4, label="Krzywa ROC (AUC = %0.2f)" % auc_pr)
    plt.plot([0, 1], [1, 0], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Czułość")
    plt.ylabel("Precyzja")
    plt.title("Krzywa Precyzja-Czułość")
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.savefig("plots/wykres_roc_pr.png")
