# Will Keesler
# CS760 Spring 2023
import argparse
import logging

from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

from hw2 import DecisionTreeClassifier, parse_data_file, show_tree


_logger = logging.getLogger("HW2_questions")

def q2_1():
    pass

def q2_2():
    pass

def q2_3():
    _logger.info("Question 2.3")
    data = parse_data_file(Path("data/Druns.txt"))
    _logger.info("Number of observations: %s", len(data))
    root_node = DecisionTreeClassifier(data, log_info=True).root_node
    _logger.info(root_node)
    show_tree(root_node)

def q2_4():
    _logger.info("Question 2.4")
    data = parse_data_file(Path("data/D3leaves.txt"))
    _logger.info("Number of observations: %s", len(data))
    root_node = DecisionTreeClassifier(data).root_node
    _logger.info(root_node)
    show_tree(root_node)

def q2_5():
    _logger.info("Question 2.5")
    data = parse_data_file(Path("data/D1.txt"))
    _logger.info("Number of observations: %s", len(data))
    root_node = DecisionTreeClassifier(data).root_node
    _logger.info(root_node)
    show_tree(root_node)
    data = parse_data_file(Path("data/D2.txt"))
    _logger.info("Number of observations: %s", len(data))
    root_node = DecisionTreeClassifier(data).root_node
    _logger.info(root_node)
    show_tree(root_node)

def q2_6():
    _logger.info("Question 2.6")
    data = parse_data_file(Path("data/D1.txt"))
    classifier = DecisionTreeClassifier(data)
    classifier.plot_tree("q_2_6_d1.png")
    data = parse_data_file(Path("data/D2.txt"))
    classifier = DecisionTreeClassifier(data)
    classifier.plot_tree("q_2_6_d2.png")

def _q2_7(train_set, test_set, test_name):
    _logger.info("Question 2.7")
    _logger.info(test_name)
    classifier = DecisionTreeClassifier(train_set)
    error_rate = classifier.test(test_set)
    num_nodes = classifier.num_nodes()
    _logger.info(f"{test_name} #Nodes {num_nodes} Error {error_rate:2.4f}")
    classifier.plot_tree(f"{test_name}.png")
    return error_rate


def q2_7():
    data = parse_data_file(Path("data/Dbig.txt"))
    n = [8192, 2048, 512, 128, 32]
    error_rates = []
    # use this test set for all the models
    d_8192, test_set = train_test_split(data, train_size=8192, random_state=32)
    err = _q2_7(d_8192, test_set, "D8192")
    error_rates.append(err)
    d_2048 = train_test_split(d_8192, train_size=2048, random_state=32)
    err = _q2_7(d_2048[0], test_set, "D2048")
    error_rates.append(err)
    d_512 = train_test_split(d_2048[0], train_size=512, random_state=32)
    err = _q2_7(d_512[0], test_set, "D512")
    error_rates.append(err)
    d_128 = train_test_split(d_512[0], train_size=128, random_state=32)
    err = _q2_7(d_128[0], test_set, "D128")
    error_rates.append(err)
    d_32 = train_test_split(d_128[0], train_size=32, random_state=32)
    err = _q2_7(d_32[0], test_set, "D32")
    error_rates.append(err)
    plt.clf()
    plt.plot(n, error_rates)
    plt.xlabel("n")
    plt.ylabel("err_n")
    plt.savefig("q_2_7_err_n.png")


def run():
    q2_1()
    q2_2()
    q2_3()
    q2_4()
    q2_5()
    q2_6()
    q2_7()


if __name__ == "__main__":
    logging.basicConfig(filename='hw2_2.log', filemode='w', encoding='utf-8', level=logging.INFO)
    try:
        run()
    except Exception as e:
        _logger.exception("Something went wrong.")
