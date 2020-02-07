#!/usr/bin/env python3

import sys
import time
import numpy as np

from PIL import Image
from pathlib import Path
from collections import defaultdict
from click import echo, style

from typing import List


# timeit: decorator to time functions
def timeit(f):
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        if "log_time" in kwargs:
            name = kwargs.get("log_name", f.__name__.upper())
            kwargs["log_time"][name] += int((te - ts) * 1000)

        else:
            echo(style(f"[DEBUG] {f.__name__}  {((te - ts) * 1000):.2f} ms"))

        return result

    return timed


def calculate_histogram(img_array: np.array, log_time=None):
    pass


def select_color(img_array: np.array, log_time=None) -> np.array:
    pass


def season(img_arr: np.array, log_time=None) -> np.array:
    pass


def gaussian_noise(img_arr: np.array, log_time=None) -> np.array:
    pass


def linear_filter(img_arr: np.array, log_time=None) -> np.array:
    pass


def median_filter(img_arr: np.array, log_time=None) -> np.array:
    pass


@timeit
def get_image_data(filename: Path, log_time=None) -> np.array:
    with Image.open(filename) as img:
        echo(
            style("[INFO] ", fg="green")
            + f"extracting data from: {style(str(filename), fg='cyan')}"
        )
        return np.array(img)


def main(argv: List[str]):

    base_path = Path(argv[1])
    files = list(base_path.glob("*.jpg"))

    time_data = defaultdict(int)

    for f in files:
        img = get_image_data(f, log_time=time_data)

        # echo(style("[INFO] ", fg="green") + f"image data: {type(img)}")

    for k, v in time_data.items():
        echo(
            style("[INFO] ", fg="green")
            + "average time data: "
            + style(f"{k} : {(v / len(files)):.2f} ms", bold=True, fg="red")
        )


if __name__ == "__main__":
    main(sys.argv)
