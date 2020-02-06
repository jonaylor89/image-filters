#!/usr/bin/env python3

import sys
import numpy as np

from PIL import Image
from pathlib import Path

from typing import List


def get_image_data(filename: Path) -> np.array:
    with Image.open(filename) as img:
        print("[INFO] extracting data from:", filename)
        return np.array(img)


def main(argv: List[str]):

    base_path = Path(argv[1])

    for f in base_path.glob("*.jpg"):
        img = get_image_data(f)

        print("[INFO] image data: ", img)


if __name__ == "__main__":
    main(sys.argv)
