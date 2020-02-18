
# CMSC 630 Image Analysis Project


- Serially process images
    - Read in image
    - Convert to numpy array
    - Apply filters
        - Salt and pepper
        - Gaussian noise
        - Convert to gray scale
    - Output histograms
    - Display performance
- Display performance for batch


```python
@jit((nd3type, nd3type, nd4type))
def nbcorr_prefetching(imgs, filters, output):
    n_imgs, n_rows, n_cols = imgs.shape
    n_filters, height, width = filters.shape

    for ii in range(n_imgs):
        for rr in range(n_rows - height + 1):
            for cc in range(n_cols - width + 1):
                for hh in range(height):
                    for ww in range(width):
                        for ff in range(n_filters):
                            imgval = imgs[ii, rr + hh, cc + ww]
                            filterval = filters[ff, hh, ww]
                            output[ii, ff, rr, cc] += imgval * filterval

@jit((nd3type, nd3type, nd4type))
def nbcorr(imgs, filters, output):
    n_rows, n_cols, n_imgs = imgs.shape
    height, width, n_filters = filters.shape

    for ii in range(n_imgs):
        for rr in range(n_rows - height + 1):
            for cc in range(n_cols - width + 1):
                for hh in range(height):
                    for ww in range(width):
                        for ff in range(n_filters):
                            imgval = imgs[rr + hh, cc + ww, ii]
                            filterval = filters[hh, ww, ff]
                            output[rr, cc, ii, ff] += imgval * filterval

def test_numba(A, B, C, prefetching=False):
    if prefetching:
        nbcorr_prefetching(A, B, C)
    else:
        nbcorr(A, B, C)
```

```sh
docker run -it -v $HOME/Repos/CMSC630_Project_1/datasets:/app/datasets jonaylor/cmsc_project_1
```
