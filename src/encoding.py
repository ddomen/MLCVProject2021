import numpy as np


def period_arc_cos(x):
    return np.arccos(rescaling(x))


def rescaling(x):
    return ((x - max(x)) + (x - min(x))) / (max(x) - min(x))


def gasf(x):
    return np.array([[np.cos(i + j) for j in period_arc_cos(x)] for i in period_arc_cos(x)])

def gadf(x):
    return np.array([[np.cos(i - j) for j in period_arc_cos(x)] for i in period_arc_cos(x)])