import numpy as np


def period_arc_cos(x, period=1, pixels=20):
    start = period - 1
    stop = (pixels + 1) * period
    return np.arccos(x[start:stop:period])


def gasf(x, period=1, pixels=20):
    return [[np.cos(i + j) for j in period_arc_cos(x, period, pixels)] for i in period_arc_cos(x, period, pixels)]


def gadf(x, period=1, pixels=20):
    return [[np.cos(i - j) for j in period_arc_cos(x, period, pixels)] for i in period_arc_cos(x, period, pixels)]