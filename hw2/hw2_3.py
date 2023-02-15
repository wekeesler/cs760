# Will Keesler
# CS760 Spring 2023
import logging

from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from hw2 import parse_data_file


_logger = logging.getLogger("HW2_3")


def _q2_3(train_set, test_set, test_name):
    _logger.info(test_name)
    # the defaults don't restrict the size of the tree
    classifier = DecisionTreeClassifier(random_state=32, criterion="entropy")
    train_X = [[x0, x1] for x0, x1, y in train_set]
    train_y = [y for x0, x1, y in train_set]
    test_X = [[x0, x1] for x0, x1, y in test_set]
    test_y = [y for x0, x1, y in test_set]
    classifier.fit(train_X, train_y)
    y_predict = classifier.predict(test_X)
    error = 0
    num_nodes = classifier.tree_.node_count
    for prediction, actual in zip(y_predict, test_y):
        error += (bool(prediction) is not actual)
    error_rate = error/len(test_set)
    _logger.info(f"{test_name} #Nodes {num_nodes} Error {error_rate:2.4f}")
    return error_rate


def run():
    data = parse_data_file(Path("data/Dbig.txt"))
    n = [8192, 2048, 512, 128, 32]
    error_rates = []
    d_8192, test_set = train_test_split(data, train_size=8192, random_state=32)
    err = _q2_3(d_8192, test_set, "D8192")
    error_rates.append(err)
    d_2048 = train_test_split(d_8192, train_size=2048, random_state=32)
    err = _q2_3(d_2048[0], test_set, "D2048")
    error_rates.append(err)
    d_512 = train_test_split(d_2048[0], train_size=512, random_state=32)
    err = _q2_3(d_512[0], test_set, "D512")
    error_rates.append(err)
    d_128 = train_test_split(d_512[0], train_size=128, random_state=32)
    err = _q2_3(d_128[0], test_set, "D128")
    error_rates.append(err)
    d_32 = train_test_split(d_128[0], train_size=32, random_state=32)
    err = _q2_3(d_32[0], test_set, "D32")
    error_rates.append(err)
    plt.clf()
    plt.plot(n, error_rates)
    plt.xlabel("n")
    plt.ylabel("err_n")
    plt.savefig("q_2_3_err_n.png")


if __name__ == "__main__":
    logging.basicConfig(filename='hw2_3.log', filemode='w', encoding='utf-8', level=logging.INFO)
    try:
        run()
    except Exception as e:
        _logger.exception("Something went wrong.")
