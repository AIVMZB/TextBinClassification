from datapreparation import get_train_data
from model import train_model


def main():
    x_train, y_train = get_train_data()
    train_model(x_train, y_train)


if __name__ == '__main__':
    main()
