from pathlib import Path
from typing import Literal

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import sentiment_analysis_bow as bow
import sentiment_analysis_st as st
import utils

def sentiment_analysis_validaton(
    csv_file_path: Path,
    examples_number_for_validation: int,
    n_repeats: int,
    K: int,
    transformer_type: Literal["bow", "st"],
) -> None:
    if transformer_type == "bow":
        data_matrix = bow.get_data_matrix(csv_file_path, examples_number_for_validation)
    elif transformer_type == "st":
        data_matrix = st.get_data_matrix(csv_file_path, examples_number_for_validation)

    models_metrics = utils.get_models_metrics(data_matrix, n_repeats, K, transformer_type)
    utils.show_models_metrics(models_metrics, utils.MODELS.keys(), transformer_type, "accuracy")
    utils.show_models_metrics(models_metrics, utils.MODELS.keys(), transformer_type, "f1_score")
    utils.show_models_metrics(models_metrics, utils.MODELS.keys(), transformer_type, "precision")
    utils.show_models_metrics(models_metrics, utils.MODELS.keys(), transformer_type, "recall")


def sentiment_analysis_test(
    csv_file_path: Path, model_name: str, transformer_type: Literal["bow", "st"]
) -> None:
    if transformer_type == "bow":
        data_matrix = bow.get_data_matrix(csv_file_path)
    elif transformer_type == "st":
        data_matrix = st.get_data_matrix(csv_file_path)

    X = data_matrix[:, :-1]
    Y = data_matrix[:, -1]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print(f"Test of model: {model_name}")
    model = utils.MODELS[model_name]
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    accuracy = accuracy_score(Ytest, Ypred)
    f1score = f1_score(Ytest, Ypred, average="macro")
    precision = precision_score(Ytest, Ypred, average="macro")
    recall = recall_score(Ytest, Ypred, average="macro")
    print(f"Accuracy: {accuracy}")
    print(f"F1_score: {f1score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    
    
if __name__ == "__main__":
    csv_file_path = Path("./TheSocialDilemma.csv")
    examples_number_for_validation = 5_000
    n_repeats = 2
    K = 5
    sentiment_analysis_validaton(
        csv_file_path, examples_number_for_validation, n_repeats, K, "bow"
    )
    sentiment_analysis_validaton(
        csv_file_path, examples_number_for_validation, n_repeats, K, "st"
    )
    # Random Forest algorithm with Bag Of Words transformer had the best results 
    sentiment_analysis_test(csv_file_path, "RF", "bow")
