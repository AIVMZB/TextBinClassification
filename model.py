import numpy as np
import tensorflow as tf
from datapreparation import Config


def create_model(c: Config):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(c.VOCAB_SIZE, 64, input_length=c.MAX_LENGTH),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(x: np.ndarray, y: np.ndarray, c: Config):
    model = create_model(c)
    history = model.fit(x, y, epochs=3, validation_split=0.2)
    model.save("trained_model")

    return history
