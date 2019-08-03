import numpy as np

float32 = np.float32
float64 = np.float64

zeros = np.zeros
ones = np.ones


def random_normal(shape):
    return np.random.normal(size=shape)