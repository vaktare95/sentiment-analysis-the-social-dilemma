from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


MODELS = {
    "ADA": AdaBoostClassifier(),
    "LR": LogisticRegression(),
    "KN": KNeighborsClassifier(),
    "MLP": MLPClassifier(),
    "RF": RandomForestClassifier(),
    "SVM": SVC(decision_function_shape="ovo"),
}


def get_csv_data(
    csv_file_path: Path,
    examples_number: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_data = pd.read_csv(csv_file_path, sep=",")
    if examples_number:
        csv_data = csv_data.iloc[:examples_number]
    csv_data = csv_data[["text", "Sentiment"]]
    positive_data = csv_data.loc[csv_data.Sentiment == "Positive", "text"]
    neutral_data = csv_data.loc[csv_data.Sentiment == "Neutral", "text"]
    negative_data = csv_data.loc[csv_data.Sentiment == "Negative", "text"]
    return positive_data, neutral_data, negative_data


def convert_csv_to_list(csv_data: pd.DataFrame) -> List[str]:
    return csv_data.values.T.tolist()


def get_models_metrics(
    data_matrix: np.ndarray,
    n_repeats: int,
    K: int,
    transformer_type: Literal["bow", "st"],
) -> Dict[str, np.ndarray]:
    kf = KFold(n_splits=K, shuffle=True)

    models_number = len(MODELS)
    models_metrics = {
        "accuracy": np.empty([models_number, n_repeats * K]),
        "f1_score": np.empty([models_number, n_repeats * K]),
        "precision": np.empty([models_number, n_repeats * K]),
        "recall": np.empty([models_number, n_repeats * K]),
    }
    m = 1
    for model_name, model in MODELS.items():
        print(
            f"Validation of model {model_name}: {m}/{models_number} model with {transformer_type} transformer"
        )
        for r in range(n_repeats):
            k = 1
            print(f"Repeat: {r+1}/{n_repeats}")
            for train_index, test_index in kf.split(data_matrix):
                print(f"Validation: {k}/{K}")
                Xtrain = data_matrix[train_index, :-1]
                Ytrain = data_matrix[train_index, -1]
                Xtest = data_matrix[test_index, :-1]
                Ytest = data_matrix[test_index, -1]

                model.fit(Xtrain, Ytrain)
                Ypred = model.predict(Xtest)
                accuracy = accuracy_score(Ytest, Ypred)
                f1score = f1_score(Ytest, Ypred, average="macro")
                precision = precision_score(Ytest, Ypred, average="macro")
                recall = recall_score(Ytest, Ypred, average="macro")

                models_metrics["accuracy"][m - 1, r * K + k - 1] = accuracy
                models_metrics["f1_score"][m - 1, r * K + k - 1] = f1score
                models_metrics["precision"][m - 1, r * K + k - 1] = precision
                models_metrics["recall"][m - 1, r * K + k - 1] = recall
                k += 1
            r += 1
        m += 1

    return models_metrics


def show_models_metrics(
    models_metrics: Dict[str, np.ndarray],
    models_names: List[str],
    transformer_type: Literal["bow", "st"],
    metric_type: Literal["accuracy", "f1_score", "precision", "recall"],
) -> None:
    _, ax = plt.subplots()
    ax.boxplot(models_metrics[metric_type].T)
    plt.xticks(np.arange(len(models_names)) + 1, models_names)
    plt.ylabel(metric_type)
    title = f"{transformer_type} transformer - models {metric_type}"
    plt.title(title)
    plt.savefig(f"./{title}.png")
