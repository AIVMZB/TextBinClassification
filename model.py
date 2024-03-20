import numpy as np
import tensorflow as tf
import config as c


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(c.VOCAB_SIZE, 64, input_length=c.MAX_LENGTH),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(x: np.ndarray, y: np.ndarray):
    model = create_model()
    history = model.fit(x, y, epochs=2, validation_split=0.2)
    model.save("trained_model")

    return history
