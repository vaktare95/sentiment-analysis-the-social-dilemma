from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import utils


ST_MODEL = SentenceTransformer("all-mpnet-base-v2")


def get_tokenized_tweets(tweets: List[str]) -> List[str]:
    tokenized_tweets = []
    for tweet in tweets:
        tokens = tweet.split()
        tokens = [t for t in tokens if not any([c in t for c in ["@", "…", "https"]])]
        tokenized_tweets.append(" ".join(tokens).strip())
    return tokenized_tweets


def create_data_matrix_from_tweets(
    positives_tokenized: List[str],
    neutrals_tokenized: List[str],
    negatives_tokenized: List[str],
) -> np.ndarray:
    data_matrix_positives = ST_MODEL.encode(positives_tokenized)
    V = data_matrix_positives.shape[0]
    data_matrix_positives = np.concatenate(
        (data_matrix_positives, 3 * np.ones((V, 1))), axis=1
    )

    data_matrix_neutrals = ST_MODEL.encode(neutrals_tokenized)
    V = data_matrix_neutrals.shape[0]
    data_matrix_neutrals = np.concatenate(
        (data_matrix_neutrals, 2 * np.ones((V, 1))), axis=1
    )

    data_matrix_negatives = ST_MODEL.encode(negatives_tokenized)
    V = data_matrix_negatives.shape[0]
    data_matrix_negatives = np.concatenate(
        (data_matrix_negatives, np.ones((V, 1))), axis=1
    )

    return np.concatenate(
        (data_matrix_positives, data_matrix_neutrals, data_matrix_negatives),
        axis=0,
    )


def get_data_matrix(
    csv_file_path: Path, examples_number_for_validation: Optional[int] = None
) -> np.ndarray:
    if examples_number_for_validation:
        positive_data, neutral_data, negative_data = utils.get_csv_data(
            csv_file_path, examples_number_for_validation
        )
    else:
        positive_data, neutral_data, negative_data = utils.get_csv_data(csv_file_path)

    positive_tweets = utils.convert_csv_to_list(positive_data)
    neutral_tweets = utils.convert_csv_to_list(neutral_data)
    negative_tweets = utils.convert_csv_to_list(negative_data)

    positives_tokenized = get_tokenized_tweets(positive_tweets)
    neutrals_tokenized = get_tokenized_tweets(neutral_tweets)
    negatives_tokenized = get_tokenized_tweets(negative_tweets)

    return create_data_matrix_from_tweets(
        positives_tokenized, neutrals_tokenized, negatives_tokenized
    )
