import numpy as np


def gasf(x, period=1, pixels=20):
    start = period - 1
    stop = (pixels + 1) * period
    x = x[start:stop:period]
    max_x, min_x = max(x), min(x)
    x_tilde = [((i - max_x) + (i - min_x)) / (max_x - min_x) for i in x]
    teta = np.arcsin(x_tilde)
    result = [[np.cos(i + j) for j in teta] for i in teta]

    return np.array(result)


def gadf(x, period=1, pixels=20):
    start = period - 1
    stop = (pixels + 1) * period
    x = x[start:stop:period]
    max_x, min_x = max(x), min(x)
    x_tilde = [((i - max_x) + (i - min_x)) / (max_x - min_x) for i in x]
    teta = np.arcsin(x_tilde)
    result = [[np.cos(i - j) for j in teta] for i in teta]

    return np.array(result)
