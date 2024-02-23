from config import Config
from datapreparation import preprocess
from model import train_model


def main():
    c = Config()
    x_train, y_train, x_test, y_test = preprocess(c)
    train_model(x_train, y_train, c)


if __name__ == '__main__':
    main()
