# Will Keesler
# CS760 Spring 2023
import logging
import numpy as np

from matplotlib import pyplot as plt
from scipy.interpolate import lagrange

_logger = logging.getLogger("HW2_4")


def _interpolate(train_x, train_y, test_x, test_y, sigma, rng):
    train_x += rng.normal(0, sigma, len(train_x))
    poly_fit = lagrange(train_x, train_y)

    train_error = np.linalg.norm(poly_fit(train_x) - train_y, ord=2)
    test_error = np.linalg.norm(poly_fit(test_x) - test_y, ord=2)
    return train_error, test_error


def run():
    a = 0
    b = 10
    # array of standard deviations for noise, include zero
    sigmas = np.linspace(0, 2*np.pi, 20)
    rng = np.random.default_rng(12345)
    train_x = rng.uniform(a, b, 17)
    _logger.info(train_x)
    train_y = np.sin(train_x)
    test_x = rng.uniform(a, b, 100)
    test_y = np.sin(test_x)
    train_errors = []
    test_errors = []
    for sigma in sigmas:
        train_error, test_error = _interpolate(train_x, train_y, test_x, test_y, sigma, rng)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.clf()
    plt.plot(sigmas, train_errors, 'b-', label="Train Error")
    plt.plot(sigmas, test_errors, 'r-', label="Test Error")
    plt.yscale("log")
    plt.xlabel("Sigma")
    plt.ylabel("log(Error)")
    plt.legend()
    plt.savefig("q_2_4_err.png")


if __name__ == "__main__":
    logging.basicConfig(filename='hw2_4.log', filemode='w', encoding='utf-8', level=logging.INFO)
    try:
        run()
    except Exception as e:
        _logger.exception("Something went wrong.")
