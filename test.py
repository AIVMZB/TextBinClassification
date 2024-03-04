import keras
from keras import Model
from sklearn.metrics import classification_report

from config import Config
import datapreparation as dp
from sklearn.metrics import confusion_matrix


def get_model() -> Model:
    return keras.models.load_model("trained_model")


def get_model_prediction(text: str, model: Model, config: Config):
    tokenized_text = dp.tokenize([text], config)

    prediction = model.predict([tokenized_text])

    return prediction


def evaluate_model(x_test, y_test):
    model: keras.models.Model = keras.models.load_model("trained_model")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"[Test loss] - {test_loss}, [Test accuracy] - {test_acc}")

    pp = model.predict(x_test)

    y_pred = pp.round()

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    cr = classification_report(y_test, y_pred)
    print(cr)


if __name__ == '__main__':
    config = Config()
    x_test, y_test = dp.get_test_data(config)

    evaluate_model(x_test, y_test)
