import numpy as np
import pandas as pd


def vectorize_seq(seq: np.ndarray, dimension=10000):
    results = np.zeros((len(seq), dimension))
    for i, seq in enumerate(seq):
        results[i, seq] = 1
    return results


def main():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    x_train = vectorize_seq(train_data)
    y_train = vectorize_seq(test_data)


if __name__ == '__main__':
    main()

