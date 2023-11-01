

import numpy as np
from .histogram import hist_normalize
from .metrics import cosine_similarity


def stat_similarity(img1:np.ndarray, img2:np.ndarray):
    hist1 = hist_normalize(img1).reshape(-1)
    hist2 = hist_normalize(img2).reshape(-1)
    
    v1 = stat_feature_vector(hist1).reshape(1,-1)
    v2 = stat_feature_vector(hist2).reshape(1,-1)

    return cosine_similarity(v1, v2)


def stat_feature_vector(hist):
    return np.array([
        mean(hist),
        n_moments(hist, 2), # variance
        n_moments(hist, 3), # skewness
        n_moments(hist, 4), # kurtosis
        entropy(hist),
        energy(hist)
    ])

def mean(hist):
    levels = 256
    intensities = (np.arange(levels) - (levels // 2 - 1)) / (levels // 2) # normalization to [-1, 1]
    mean_intensity = (intensities * hist)
    mean_intensity = mean_intensity.sum()
    return mean_intensity


def n_moments(hist, n):
    levels = 256
    intensities = (np.arange(levels) - (levels // 2 - 1)) / (levels // 2) # normalization to [-1, 1]
    n_moment = ((intensities - mean(hist)) ** n * hist).sum()
    return n_moment


def entropy(hist):
    # add epsilon to hist so that log2(0) = 0
    hist = hist + np.finfo(float).eps
    entropy = - (hist * np.log2(hist)).sum()
    return (entropy - 4) / 4


def energy(hist):
    energy = (hist ** 2).sum() * 2 - 1
    return energy
