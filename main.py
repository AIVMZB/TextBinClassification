from config import Config
from datapreparation import get_train_data
from model import train_model


def main():
    c = Config()
    x_train, y_train = get_train_data(c)
    train_model(x_train, y_train, c)


if __name__ == '__main__':
    main()
