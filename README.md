# CMSC 630 Project 1

## Overview

### Execution

    pip3 install --user pipenv
    pipenv install
    pipenv run python main.py

Or

    docker run -it -v $HOME/Repos/CMSC630_Project_1/datasets:/app/datasets jonaylor/cmsc_project_1

*(this should pull the image already on dockerhub so the docker image won't have to be build locally)*

## Implementation

- Python
    - Numpy
        - Numpy arrays
    - Numba
        - Jit
        - Fastmath
    - Matplotllib
        - Histogram
    - Pillow
        - Reading and writing image data
- Dependency List


    [packages]
    pillow = "*"
    numpy = "*"
    matplotlib = "*"
    click = "**"
    numba = "*"
    toml = "*"
    tqdm = "*"

- Multiprocessing batches
- Function decorators for time
- Toml configuration `config.toml`

## Functions

    calculate_histogram(img_array: np.array) -> np.array

    mean_square_error(original_img: np.array, quantized_img: np.array) -> int

    select_channel(img_array: np.array, color: str = "", log_time=None) -> np.array

    salt_pepper_noise(img_array: np.array, strength: int) -> np.array

    gaussian_noise(img_array: np.array, sigma: int) -> np.array

    linear_filter(img_array: np.array, mask_size: int, weights: List[List[int]]) -> np.array

    median_filter(img_array: np.array, mask_size: int, weights: List[List[int]]) -> np.array

    apply_operations(img_file: Path)

    parallel_operations(files: List[Path])

## Results

![CMSC%20630%20Project%201/Screen_Shot_2020-03-08_at_12.02.26_PM.png](assets/Screen_Shot_2020-03-08_at_12.02.26_PM.png)

![CMSC%20630%20Project%201/Screen_Shot_2020-03-08_at_12.02.08_PM.png](assets/Screen_Shot_2020-03-08_at_12.02.08_PM.png)
