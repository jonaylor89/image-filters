#!/usr/bin/env python3

import sys
import time
import numpy as np

from PIL import Image
from pathlib import Path
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
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            echo(style(f"[DEBUG] {f.__name__}  {((te - ts) * 1000):.2f} ms"))

        return result

    return timed


def calculate_histogram(img_array: np.array):
    pass


def select_color(img_array: np.array) -> np.array:
    pass


def season(img_arr: np.array) -> np.array:
    pass


def gaussian_noise(img_arr: np.array) -> np.array:
    pass


def linear_filter(img_arr: np.array) -> np.array:
    pass


def median_filter(img_arr: np.array) -> np.array:
    pass


@timeit
def get_image_data(filename: Path) -> np.array:
    with Image.open(filename) as img:
        echo(style("[INFO] ", fg="green") + f"extracting data from: {filename}")
        return np.array(img)


def main(argv: List[str]):

    base_path = Path(argv[1])

    for f in base_path.glob("*.jpg"):
        img = get_image_data(f)

        echo(style(f"[INFO] ", fg="green") + f"image data: {type(img)}")


if __name__ == "__main__":
    main(sys.argv)
