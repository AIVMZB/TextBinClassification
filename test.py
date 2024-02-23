import pickle

import keras
import numpy as np
import pandas as pd
from keras import Model
from keras.utils import pad_sequences
from sklearn.metrics import classification_report

from config import Config
import datapreparation as dp


def tokenize_text(text: str, config: Config) -> np.ndarray:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        seq = tokenizer.texts_to_sequences([text])

        padded = pad_sequences(seq, maxlen=config.MAX_LENGTH,
                               padding=config.PADDING_TYPE, truncating=config.TRUNC_TYPE)

        return np.array(padded)


def get_model() -> Model:
    return keras.models.load_model("trained_model")


def get_model_prediction(text: str, model: Model, config: Config):
    tokenized_text = tokenize_text(text, config)

    prediction = model.predict([tokenized_text])

    return prediction


def evaluate_model(x_test, y_test):
    model: keras.models.Model = keras.models.load_model("trained_model")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"[Test loss] - {test_loss}, [Test accuracy] - {test_acc}")

    pp = model.predict(x_test)
    y_pred = np.argmax(pp, axis=1)

    pd.crosstab(y_test, y_pred, rownames=["True"], colnames=["Predicted"], margins=True)
    cr = classification_report(y_test, y_pred)
    print(cr)


if __name__ == '__main__':
    config = Config()
    x_train, y_train, x_test, y_test = dp.preprocess(config)

    evaluate_model(x_test, y_test)
