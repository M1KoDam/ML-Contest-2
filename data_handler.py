import numpy as np
import pandas as pd
from collections import defaultdict


def num_seq(seq):
    words = defaultdict(int)
    for row in seq:
        for word in row:
            words[word] += 1
    print(len(words))


def vectorize_seq(seq, dimension=10000):
    results = np.zeros((len(seq), dimension))
    for i, new_seq in enumerate(seq):
        print(new_seq)
        results[i, new_seq] = 1
    return results


def main():
    train_data_p = pd.read_csv('data/train.csv')
    test_data_p = pd.read_csv('data/test.csv')

    train_data = train_data_p["text"].to_numpy()
    train_labels = train_data_p["rating"].to_numpy()

    seq = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    vectorized_seq = vectorize_seq(seq, dimension=10)

    print(vectorized_seq)
    num_seq(train_data)
    v = []
    for td in train_data:
        v.append(np.array(td.split()))

    x_train = vectorize_seq(v)
    # y_train = vectorize_seq(train_labels)

    print(x_train)


if __name__ == '__main__':
    main()

