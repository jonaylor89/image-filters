#!/usr/bin/env python3

import sys
import time
import numpy as np

from PIL import Image
from pathlib import Path
from click import echo, style
import matplotlib.pyplot as plt
from collections import defaultdict

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
            echo(
                style(
                    f"[DEBUG] {f.__name__}  {((te - ts) * 1000):.2f} ms",
                    bold=True,
                    fg="pink",
                )
            )

        return result

    return timed


def calculate_histogram(img_array: np.array, log_time=None):
    """
    >> h=zeros(256,1);              OR    >> h=zeros(256,1);
    >> for l = 0 : 255                    >> for l = 0 : 255
         for i = 1 : N                          h(l +1)=sum(sum(A == l));
            for j = 1 : M                    end
                if (A(i,j) == l)          >> bar(0:255,h);
                    h(l +1) = h(l +1) +1;
                end
            end
        end
    end
    >> bar(0:255,h);
    """
    
    # Get the size of the image
    N, M = img_array.shape

    # Create blank histpgram
    hist = np.zeros(256)

    for l in range(256):
        for i in range(N):
            for j in range(M):

                # Loop through pixels to calculate histogram
                if img_arr[i][j] == 1:
                    h[l+1] += 1

    # Plot histogram
    _ = plt.hist(a, bins='auto')
    plt.title("Histogram with 'auto' bins")
    plt.show()


def histrogram_equalization(img_array: np.array, log_time=None):
    """
    g1(l) = ∑(l, k=0) pA(k) ⇒ g1(l)−g1(l −1) = pA(l) = hA(l)/NM (l = 1,...,255)

    geA(l) = round(255g1(l))
    """
    pass


def select_color(img_array: np.array, color: str, log_time=None) -> np.array:
    pass


@timeit
def season(img_arr: np.array, strength: int, log_time=None) -> np.array:
    pass


@timeit
def gaussian_noise(img_arr: np.array, params: int, log_time=None) -> np.array:
    pass


@timeit
def linear_filter(
    img_arr: np.array, mask_size: int, weights: List[List[int]], log_time=None
) -> np.array:
    pass


@timeit
def median_filter(
    img_arr: np.array, mask_size: int, weights: List[List[int]], log_time=None
) -> np.array:
    pass


def export_image(img_arr: np.array, filename: str) -> None:
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
    echo(style("[INFO] ", fg="green") + f"image directory: {str(base_path)}")

    files = list(base_path.glob("*.BMP"))

    time_data = defaultdict(int)

    for f in files:
        img = get_image_data(f, log_time=time_data)

        # echo(style("[INFO] ", fg="green") + f"image data: {type(img)}")
        #
        salt_and_pepper = season(img, log_time=time_data)
        export_image(salt_and_pepper, "salt_and_pepper_" + f)

        guass = gaussian_noise(img, log_time=time_data)
        export_image(guass, "guassian_" + f)

        linear = linear_filter(img, log_time=time_data)
        export_image(linear, "linear_" + f)

        median = median_filter(img, log_time=time_data)
        export_image(median, "median_" + f)

        # calculate_histogram(img, log_time=time_data)
        # histrogram_equalization(img, log_time=time_data)

    for k, v in time_data.items():
        echo(
            style("[INFO] ", fg="green")
            + "average time data: "
            + style(f"{k} : {(v / len(files)):.2f} ms", bold=True, fg="red")
        )


if __name__ == "__main__":
    main(sys.argv)
