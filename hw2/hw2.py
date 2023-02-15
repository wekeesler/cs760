# Will Keesler
# CS760 Spring 2023
import argparse
import logging
import math
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path

_logger = logging.getLogger("HW2")


def apply_split(data, split):
    feature_index, threshold = split
    left = []
    right = []
    for vals in data:
        if vals[feature_index] >= threshold:
            left.append(vals)
        else:
            right.append(vals)
    return left, right


def compute_class_counts(data):
    true_count = 0
    false_count = 0
    for vals in data:
        if vals[-1]:
            true_count += 1
        else:
            false_count += 1
    return true_count, false_count


def determine_class_label(data):
    true_count, false_count = compute_class_counts(data)
    # majority vote
    if true_count > false_count:
        return True
    elif false_count > true_count:
        return False
    # return True in case of tie
    return True


def compute_entropy(total, class_counts):
    entropy = 0
    # empty node has zero entropy
    if total == 0:
        return entropy
    for class_count in class_counts:
        # class count of zero has zero entropy
        if class_count > 0:
            p_class = class_count/total
            entropy -= p_class*math.log2(p_class)
    return entropy


def compute_info_gain_ratio(data, split):
    total = len(data)
    left, right = apply_split(data, split)
    split_entropy = compute_entropy(total, [len(left), len(right)])

    h_y = compute_entropy(total, compute_class_counts(data))

    h_y_left = compute_entropy(len(left), compute_class_counts(left))
    if total == 0:
        p_left = 0
    else:
        p_left = len(left)/len(data)

    h_y_right = compute_entropy(len(right), compute_class_counts(right))
    if total == 0:
        p_right = 0
    else:
        p_right = len(right)/len(data)

    info_gain = h_y - (p_left*h_y_left + p_right*h_y_right)
    if split_entropy == 0:
        info_gain_ratio = 0
    else:
        info_gain_ratio = info_gain/split_entropy

    return info_gain_ratio, split_entropy, info_gain


def find_best_split(data, splits):
    split_infos = []
    best_split = None
    best_info_gain_ratio = 0
    for split in splits:
        info_gain_ratio, split_entropy, info_gain = compute_info_gain_ratio(data, split)
        # only include splits with information gain
        if split_entropy > 0 and info_gain_ratio > 0:
            if info_gain_ratio > best_info_gain_ratio:
                best_info_gain_ratio = info_gain_ratio
                best_split = split
            split_infos.append((info_gain_ratio, split_entropy, info_gain, split))
    return best_split, split_infos


def check_stop(data, best_split, split_infos):
    # the stop conditions are satisfied if we didn't find a split
    if not best_split:
        return True

    return False


def determine_candidate_numeric_splits(data, feature_index):
    thresholds = set()
    # get all the unit feature values
    for vals in data:
        thresholds.add(vals[feature_index])
    splits = [(feature_index, threshold) for threshold in sorted(list(thresholds))]
    return splits

def determine_candidate_splits(data):
    splits = determine_candidate_numeric_splits(data, 0)
    splits += determine_candidate_numeric_splits(data, 1)
    return splits


def make_subtree(data, depth, log_info=False):
    splits = determine_candidate_splits(data)
    best_split, split_infos = find_best_split(data, splits)
    if log_info:
        # for problem 2.3
        _logger.info("Tree depth: %s", depth)
        for split_info in split_infos:
            info_gain_ratio, split_entropy, info_gain, split = split_info
            _logger.info(
                "(Feature, Threshold): %s, Split Entropy: %.4f, InfoGain: %.4f, InfoGainRatio: %.4f",
                split, split_entropy, info_gain, info_gain_ratio)
    if check_stop(data, best_split, split_infos):
        node = {
            'type': 'leaf',
            'class': determine_class_label(data),
        }
    else:
        left, right = apply_split(data, best_split)
        left_node = make_subtree(left, depth + 1)
        right_node = make_subtree(right, depth + 1)
        node = {
            'type': 'internal',
            'split': best_split,
            'left': left_node,
            'right': right_node,
        }

    return node


def show_tree(root_node):
    def show_tree_recurse(depth, node):
        prefix = "|----" * depth
        if node["type"] == "internal":
            feature_idx, threshold = node["split"]
            lines.append(f"{prefix}|if x_{feature_idx} >= {threshold}")
            lines.append(f"{prefix}|then")
            show_tree_recurse(depth + 1, node["left"])
            lines.append(f"{prefix}|else")
            show_tree_recurse(depth + 1, node["right"])
        else:
            prediction = node["class"]
            lines.append(f"{prefix}|Predict {prediction}")

    lines = ["\n"]
    show_tree_recurse(0, root_node)
    _logger.info("\n".join(lines))


class DecisionTreeClassifier:

    def __init__(self, data, log_info=False):
        self.data = data
        self.root_node = make_subtree(data, 0, log_info=log_info)

    def num_nodes(self):
        def num_nodes_recurse(node):
            n = 1
            if node["type"] == "internal":
                n += num_nodes_recurse(node["left"])
                n += num_nodes_recurse(node["right"])
            return n

        n = num_nodes_recurse(self.root_node)
        return n

    def predict(self, vals, node=None):
        if node is None:
            node = self.root_node

        if node["type"] == "leaf":
            return node["class"]
        feature_idx, threshold = node["split"]
        if vals[feature_idx] >= threshold:
            return self.predict(vals, node["left"])

        return self.predict(vals, node["right"])

    def test(self, data):
        error = 0
        for vals in data:
            prediction = self.predict(vals)
            actual = vals[-1]
            error += (prediction is not actual)
        return error/len(data)

    def show_tree(self):
        return show_tree(self.root_node)

    def plot_boundary(self, min_x, max_x, min_y, max_y):
        def plot_boundary_recurse(depth, node, min_x, max_x, min_y, max_y):
            if node["type"] == "internal":
                feature_idx, threshold = node["split"]
                if feature_idx == 0:
                    x_vals = threshold*np.ones(10)
                    y_vals = np.linspace(min_y, max_y, 10)
                    plt.plot(x_vals, y_vals, color="black", label="_nolegend_")
                    # update the bounds
                    # on the left side x_0 >= threshold
                    plot_boundary_recurse(depth+1, node["left"], threshold, max_x, min_y, max_y)
                    # on the left side x_0 < threshold
                    plot_boundary_recurse(depth+1, node["right"], min_x, threshold, min_y, max_y)
                else:
                    y_vals = threshold*np.ones(10)
                    x_vals = np.linspace(min_x, max_x, 10)
                    plt.plot(x_vals, y_vals, color="black", label="_nolegend_")
                    # update the bounds
                    # on the left side x_1 >= threshold
                    plot_boundary_recurse(depth+1, node["left"], min_x, max_x, threshold, max_y)
                    # on the left side x_1 < threshold
                    plot_boundary_recurse(depth+1, node["right"], min_x, max_x, min_y, threshold)

        plot_boundary_recurse(0, self.root_node, min_x, max_x, min_y, max_y)

    def plot_tree(self, file_name, boundary_only=False):
        plt.clf()
        tx = []
        ty = []
        fx = []
        fy = []
        for x, y, c in self.data:
            if c:
                tx.append(x)
                ty.append(y)
            else:
                fx.append(x)
                fy.append(y)
        max_x = max(max(tx), max(fx))
        max_y = max(max(ty), max(fy))
        min_x = min(min(tx), min(fx))
        min_y = min(min(ty), min(fy))
        self.plot_boundary(min_x, max_x, min_y, max_y)
        if not boundary_only:
            plt.plot(tx, ty, 'bo', label="True", ms=2)
            plt.plot(fx, fy, 'ro', label="False", ms=2)
            plt.legend()
        plt.xlabel("x_0")
        plt.ylabel("x_1")
        plt.savefig(file_name)


def parse_data_file(data_file_path):
    data = []
    with open(data_file_path) as data_file:
        for line in data_file.readlines():
            x1, x2, y = line.split()
            vals = (float(x1), float(x2), bool(float(y)))
            data.append(vals)
    return data


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    args = parser.parse_args()
    _logger.info("Data file: %s", args.data_file)
    data = parse_data_file(Path(args.data_file))
    _logger.info("Number of observations: %s", len(data))
    root_node = DecisionTreeClassifier(data).root_node
    _logger.info(root_node)
    show_tree(root_node)


if __name__ == "__main__":
    logging.basicConfig(filename='hw2.log', filemode='w', encoding='utf-8', level=logging.INFO)
    try:
        run()
    except Exception as e:
        _logger.exception("Something went wrong.")
