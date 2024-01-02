from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import nltk
import numpy as np

import utils

WORDNET_LEMMATIZER = nltk.stem.WordNetLemmatizer()

CUSTOM_STOPWORDS = {
    "thesocialdilemma",
    "movie",
    "movies",
    "film",
    "films",
    "netflix",
    "twitter",
    "tweet",
    "tweets",
    "retweet",
    "retweets",
}


def get_stopwords() -> Set[str]:
    # from http://www.lextek.com/manuals/onix/stopwords1.html
    stopwords = set(w.strip() for w in open("./stopwords.txt"))
    return stopwords.union(CUSTOM_STOPWORDS)


# custom tokenizer instead of nltk.tokenize.word_tokenize()
def tweet_text_tokenize(tweet_text: str) -> List[str]:
    return [
        "".join(c for c in token if c.isalpha() or c in ["@", "…"])
        for token in tweet_text.split()
    ]


def tokenize_tweet(tweet_text: str, stopwords: Set[str]):
    tweet_text = tweet_text.lower()
    tokens = tweet_text_tokenize(tweet_text)
    tokens = [
        t
        for t in tokens
        if len(t) > 2 and not any([c in t for c in ["@", "…", "https"]])
    ]
    tokens = [WORDNET_LEMMATIZER.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


def update_word_index_map_with_tokenized_tweets(
    word_index_map: Dict[str, int],
    tweets: List[str],
    current_index: int,
    stopwords: Set[str],
) -> Tuple[Dict[str, int], List[List[str]], int]:
    tokenized_tweets = []
    for tweet in tweets:
        tokens = tokenize_tweet(tweet, stopwords)
        tokenized_tweets.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
    return word_index_map, tokenized_tweets, current_index


def tokens_to_vector(tokens: List[str], label: int, word_index_map: Dict[str, int]):
    vector = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        vector[i] += 1
    if vector.sum():
        vector = vector / vector.sum()
    vector[-1] = label
    return vector


def create_data_matrix_from_tweets(
    tokenized_tweets: List[List[str]],
    word_index_map: Dict[str, int],
    label: int,
) -> np.ndarray:
    examples_number = len(tokenized_tweets)
    vocabulary_size = len(word_index_map)
    data_matrix = np.zeros((examples_number, vocabulary_size + 1))

    for i, tokens in enumerate(tokenized_tweets):
        xy = tokens_to_vector(tokens, label, word_index_map)
        data_matrix[i, :] = xy

    return data_matrix


def get_data_matrix(
    csv_file_path: Path, examples_number_for_validation: Optional[int] = None
) -> np.ndarray:
    stopwords = get_stopwords()

    if examples_number_for_validation:
        positive_data, neutral_data, negative_data = utils.get_csv_data(
            csv_file_path, examples_number_for_validation
        )
    else:
        positive_data, neutral_data, negative_data = utils.get_csv_data(csv_file_path)

    positive_tweets = utils.convert_csv_to_list(positive_data)
    neutral_tweets = utils.convert_csv_to_list(neutral_data)
    negative_tweets = utils.convert_csv_to_list(negative_data)

    word_index_map = {}
    current_index = 0
    (
        word_index_map,
        positives_tokenized,
        current_index,
    ) = update_word_index_map_with_tokenized_tweets(
        word_index_map,
        positive_tweets,
        current_index,
        stopwords,
    )
    (
        word_index_map,
        neutrals_tokenized,
        current_index,
    ) = update_word_index_map_with_tokenized_tweets(
        word_index_map,
        neutral_tweets,
        current_index,
        stopwords,
    )
    (
        word_index_map,
        negatives_tokenized,
        current_index,
    ) = update_word_index_map_with_tokenized_tweets(
        word_index_map,
        negative_tweets,
        current_index,
        stopwords,
    )

    data_matrix_positives = create_data_matrix_from_tweets(
        positives_tokenized, word_index_map, 3
    )
    data_matrix_neutrals = create_data_matrix_from_tweets(
        neutrals_tokenized, word_index_map, 2
    )
    data_matrix_negatives = create_data_matrix_from_tweets(
        negatives_tokenized, word_index_map, 1
    )
    return np.concatenate(
        (data_matrix_positives, data_matrix_neutrals, data_matrix_negatives), axis=0
    )
