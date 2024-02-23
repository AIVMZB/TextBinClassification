import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle
from sklearn.model_selection import train_test_split

from config import Config


def configure(config: Config, df: pd.DataFrame):
    df["words_number"] = df["review"].str.split().str.len()
    config.MAX_LENGTH = df["words_number"].max()


def get_data(config: Config) -> pd.DataFrame:
    df = pd.read_csv(config.FILE_NAME)
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    return df


def tokenize(df: pd.DataFrame, config: Config):
    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token=config.OOV_TOKEN)
    tokenizer.fit_on_texts(df["review"])
    word_index = tokenizer.word_index

    seq = tokenizer.texts_to_sequences(df["review"])

    padded = pad_sequences(seq, maxlen=config.MAX_LENGTH,
                           padding=config.PADDING_TYPE, truncating=config.TRUNC_TYPE)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return np.array(padded)


def preprocess(config: Config) -> tuple:
    """Returns (train_x, train_y, test_x, test_y)"""

    df = get_data(config)
    configure(config, df)
    train_size = int(len(df) * 0.8)
    train_df, test_df = df[:train_size], df[train_size:]

    return (tokenize(train_df, config), train_df["sentiment"].to_numpy(),
            tokenize(test_df, config), test_df["sentiment"].to_numpy())
