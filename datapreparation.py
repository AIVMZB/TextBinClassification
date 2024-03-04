import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle

from config import Config


def get_tokenizer(config: Config, load: bool = True, texts: list[str] = None) -> Tokenizer:
    if load:
        with open('tokenizer.pickle', 'rb') as handle:
            return pickle.load(handle)
    if texts is None:
        raise ValueError("You need to give texts to fit new tokenizer")

    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token=config.OOV_TOKEN)
    tokenizer.fit_on_texts(texts)

    return tokenizer


def tokenize(data: list[str], config: Config, load=True) -> np.ndarray:
    tokenizer = get_tokenizer(config, load, data)

    seq = tokenizer.texts_to_sequences(data)

    padded = pad_sequences(seq, maxlen=config.MAX_LENGTH,
                           padding=config.PADDING_TYPE, truncating=config.TRUNC_TYPE)

    return np.array(padded)


def configure(config: Config, df: pd.DataFrame):
    df["words_number"] = df["review"].str.split().str.len()
    config.MAX_LENGTH = df["words_number"].max()


def get_data(config: Config) -> pd.DataFrame:
    df = pd.read_csv(config.FILE_NAME)
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    return df


def get_train_data(config: Config) -> tuple:
    """Returns (train_x, train_y, test_x, test_y)"""

    df = get_data(config)
    configure(config, df)
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]

    x_train = tokenize(train_df["review"].tolist(), config, load=False)
    y_train = train_df["sentiment"].to_numpy()

    return x_train, y_train


def get_test_data(config: Config) -> tuple:
    df = get_data(config)
    train_size = int(len(df) * 0.8)
    test_df = df[train_size:]

    x_test = tokenize(test_df["review"].tolist(), config)
    y_test = test_df["sentiment"].to_numpy()

    return x_test, y_test
