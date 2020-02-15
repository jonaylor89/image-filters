#!/usr/bin/env python3

import sys
import time
import numpy as np

from numba import jit
from PIL import Image
from pathlib import Path
from click import echo, style

from matplotlib import pyplot as plt

from typing import List, Tuple

OUTPUT_DIR = "datasets/output/"

# timeit: decorator to time functions
def timeit(f):
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()

        echo(style("[INFO] ", fg="green") + f"{f.__name__}  {((te - ts) * 1000):.2f} ms")

        return result

    return timed


@jit(nopython=True)
def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)

    return np.array(b)


@jit(nopython=True)
def histogram(img_array):
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

    # Create blank histpgram
    hist = np.zeros(256)

    # Get size of pixel array
    N = len(img_array)

    for l in range(256):
        for i in range(N):

            # Loop through pixels to calculate histogram
            if img_array.flat[i] == l:
                hist[l] += 1

    return hist


@timeit
def calculate_histogram(img_array: np.array) -> np.array:
    """
    g1(l) = ∑(l, k=0) pA(k) ⇒ g1(l)−g1(l −1) = pA(l) = hA(l)/NM (l = 1,...,255)

    geA(l) = round(255g1(l))
    """

    flat = img_array.flatten()

    hist = histogram(flat)
    cs = cumsum(hist)

    nj = (cs - cs.min()) * 255

    N = cs.max() - cs.min()

    cs = nj / N

    cs = cs.astype("uint8")

    equalized = cs[flat]
    img_new = np.reshape(equalized, img_array.shape)

    return hist, histogram(equalized), img_new


def select_channel(img_array: np.array, color: str = "", log_time=None) -> np.array:

    if color == "red":
        return img_array[:, :, 0]

    elif color == "green":
        return img_array[:, :, 1]

    elif color == "blue":
        return img_array[:, :, 2]

    else:
        # Default to using the default greyscaling Pillow does
        img = Image.fromarray(img_array, "L")

        return np.array(img)


@timeit
@jit(nopython=True)
def season_noise(img_array, strength: int) -> np.array:
    s_vs_p = 0.5
    out = np.copy(img_array)

    # Generate Salt '1' noise
    num_salt = np.ceil(strength * img_array.size * s_vs_p)

    for i in range(int(num_salt)):
        x = np.random.randint(0, img_array.shape[0] - 1)
        y = np.random.randint(0, img_array.shape[1] - 1)
        out[x][y] = 0

    # Generate Pepper '0' noise
    num_pepper = np.ceil(strength * img_array.size * (1.0 - s_vs_p))

    for i in range(int(num_pepper)):
        x = np.random.randint(0, img_array.shape[0] - 1)
        y = np.random.randint(0, img_array.shape[1] - 1)
        out[x][y] = 0

    return out


@timeit
@jit(nopython=True)
def gaussian_noise(img_array: np.array, params: int) -> np.array:
    mean = 0.0
    var = 0.01
    sigma = var ** 0.5

    noise = np.random.normal(mean, sigma, img_array.size)
    shaped_noise = noise.reshape(img_array.shape)

    gauss = img_array + shaped_noise

    return gauss


@timeit
def linear_filter(
    img_arr: np.array, mask_size: int, weights: List[List[int]]
) -> np.array:

    # linear = apply_filter(img_size)
    pass


@timeit
def median_filter(
    img_arr: np.array, mask_size: int, weights: List[List[int]]
) -> np.array:

    # median = apply_filter(img_arr, median)
    pass


@jit(nopython=True)
def apply_filter(img_arr, img_filter):
    rows, cols = imgs.shape
    height, width = img_filter.shape

    output = np.zeros((cols - height + 1, rows - width + 1))

    for rr in range(n_rows - height + 1):
        for cc in range(n_cols - width + 1):
            for hh in range(height):
                for ww in range(width):
                    imgval = img_arr[rr + hh, cc + ww]
                    filterval = img_filter[hh, ww]
                    output[rr, cc] += imgval * filterval

    return output


def export_image(img_arr: np.array, filename: str) -> None:
    img = Image.fromarray(img_arr)
    img.save(OUTPUT_DIR + filename + ".BMP")


def export_plot(img_arr: np.array, filename: str) -> None:
    _ = plt.hist(img_arr, bins=256, range=(0, 256))
    plt.title(filename)
    plt.savefig(OUTPUT_DIR + filename + ".png")


@timeit
def get_image_data(filename: Path, log_time=None) -> np.array:
    with Image.open(filename) as img:
        return np.array(img)


def main(argv: List[str]):

    base_path = Path(argv[1])
    echo(style("[INFO] ", fg="green") + f"image directory: {str(base_path)}")

    files = list(base_path.glob("*.BMP"))

    Path("datasets/output").mkdir(parents=True, exist_ok=True)

    time_data = {}

    t0 = time.time()

    # [!!!] Only for development
    files = files[:1]

    for f in files:
        echo(
            style(f"[INFO:{f.stem}] ", fg="green")
            + f"extracting data from: {style(str(f), fg='cyan')}"
        )

        color_img = get_image_data(f, log_time=time_data)
        copy_color_img = np.copy(color_img)

        # echo(style("[INFO] ", fg="green") + f"image data: {type(img)}")

        img = select_channel(color_img, color="red")

        salt_and_pepper = season_noise(img, 0.4)

        gauss = gaussian_noise(img, 10)

        linear = linear_filter(img, 9, [[0]])

        median = median_filter(img, 9, [[0]])

        histogram, equalized, _ = calculate_histogram(img)

        echo(
            style(f"[INFO:{f.stem}] ", fg="green")
            + f"exporting plots and images for {f.stem}..."
        )

        copy_color_img[:, :, 0] = salt_and_pepper
        export_image(copy_color_img, "salt_and_pepper_" + f.stem)

        copy_color_img[:, :, 0] = gauss
        export_image(copy_color_img, "gaussian_" + f.stem)

        # export_image(linear, "linear_" + f.stem)
        # export_image(median, "median_" + f.stem)
        export_plot(histogram, "histogram_" + f.stem)
        export_plot(equalized, "histogram_equalized_" + f.stem)

    t_delta = time.time() - t0

    for k, v in time_data.items():
        echo(
            style("[INFO] ", fg="green")
            + "average time data: "
            + style(f"{k} : {(v / len(files)):.2f} ms", fg="red")
        )

    print()
    echo(style(f"[INFO] Total time: {t_delta}", fg="cyan"))


if __name__ == "__main__":
    main(sys.argv)
