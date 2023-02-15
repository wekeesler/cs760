# Will Keesler
# CS760 Spring 2023
import logging

from pathlib import Path
from sklearn.model_selection import train_test_split

from hw2 import parse_data_file

_logger = logging.getLogger("HW2_Dbig")


def run():
    data = parse_data_file(Path("data/Dbig.txt"))
    d_8192, blah = train_test_split(data, train_size=8192, random_state=32)
    d_2048, blah = train_test_split(d_8192, train_size=2048, random_state=32)
    d_512, blah = train_test_split(d_2048, train_size=512, random_state=32)
    d_128, blah = train_test_split(d_512, train_size=128, random_state=32)
    d_32, blah = train_test_split(d_128, train_size=32, random_state=32)
    with open("data/d8192.txt", "w") as f:
        lines = [f"{x1} {x2} {int(y)}\n" for x1, x2, y in d_8192]
        f.writelines(lines)
    with open("data/d2048.txt", "w") as f:
        lines = [f"{x1} {x2} {int(y)}\n" for x1, x2, y in d_2048]
        f.writelines(lines)
    with open("data/d512.txt", "w") as f:
        lines = [f"{x1} {x2} {int(y)}\n" for x1, x2, y in d_512]
        f.writelines(lines)
    with open("data/d128.txt", "w") as f:
        lines = [f"{x1} {x2} {int(y)}\n" for x1, x2, y in d_128]
        f.writelines(lines)
    with open("data/d32.txt", "w") as f:
        lines = [f"{x1} {x2} {int(y)}\n" for x1, x2, y in d_32]
        f.writelines(lines)


if __name__ == "__main__":
    logging.basicConfig(filename='hw2_dbig.log', filemode='w', encoding='utf-8', level=logging.DEBUG)
    try:
        run()
    except Exception as e:
        _logger.exception("Something went wrong.")
