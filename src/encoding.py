import numpy as np


def period_arc_cos(x):
    return np.arccos(rescaling(x))


def rescaling(x):
    return ((x - max(x)) + (x - min(x))) / (max(x) - min(x))


def gasf(x):
    period = period_arc_cos(x)
    return np.array([[np.cos(i + j) for j in period] for i in period])

def gadf(x):
    period = period_arc_cos(x)
    return np.array([[np.sin(i - j) for j in period] for i in period])