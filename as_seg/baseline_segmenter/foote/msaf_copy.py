import numpy as np
from scipy import ndimage, signal
from scipy.spatial import distance

def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in range(X.shape[1]):
        X[:, i] = ndimage.median_filter(X[:, i], size=M)
    return X


def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Foote's paper."""
    g = signal.windows.gaussian(M, M // 3.0, sym=True)
    G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
    G[M // 2 :, : M // 2] = -G[M // 2 :, : M // 2]
    G[: M // 2, M // 2 :] = -G[: M // 2, M // 2 :]
    return G


def compute_ssm(X, metric="seuclidean"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    D /= D.max()
    return 1 - D


def compute_nc(X, G):
    """Computes the novelty curve from the self-similarity matrix X and the
    gaussian kernel G."""
    N = X.shape[0]
    M = G.shape[0]
    nc = np.zeros(N)

    for i in range(M // 2, N - M // 2 + 1):
        nc[i] = np.sum(X[i - M // 2 : i + M // 2, i - M // 2 : i + M // 2] * G)

    # Normalize
    nc += nc.min()
    nc /= nc.max()
    return nc


def pick_peaks(nc, L=16):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() / 20.0

    nc = ndimage.gaussian_filter1d(nc, sigma=4)  # Smooth out nc

    th = ndimage.median_filter(nc, size=L) + offset
    # th = ndimage.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset

    peaks = []
    for i in range(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    # plt.plot(nc)
    # plt.plot(th)
    # for peak in peaks:
    # plt.axvline(peak)
    # plt.show()

    return peaks

def lognormalize(F, floor=0.1, min_db=-80):
    """Log-normalizes features such that each vector is between min_db to 0."""
    assert min_db < 0
    F = min_max_normalize(F, floor=floor)
    F = np.abs(min_db) * np.log10(F)  # Normalize from min_db to 0
    return F


def min_max_normalize(F, floor=0.001):
    """Normalizes features such that each vector is between floor to 1."""
    F += -F.min() + floor
    F = F / F.max(axis=0)
    return F


def normalize(X, norm_type, floor=0.0, min_db=-80):
    """Normalizes the given matrix of features.

    Parameters
    ----------
    X: np.array
        Each row represents a feature vector.
    norm_type: {"min_max", "log", np.inf, -np.inf, 0, float > 0, None}
        - `"min_max"`: Min/max scaling is performed
        - `"log"`: Logarithmic scaling is performed
        - `np.inf`: Maximum absolute value
        - `-np.inf`: Minimum absolute value
        - `0`: Number of non-zeros
        - float: Corresponding l_p norm.
        - None : No normalization is performed

    Returns
    -------
    norm_X: np.array
        Normalized `X` according the the input parameters.
    """
    if isinstance(norm_type, str):
        if norm_type == "min_max":
            return min_max_normalize(X, floor=floor)
        if norm_type == "log":
            return lognormalize(X, floor=floor, min_db=min_db)
    return librosa.util.normalize(X, norm=norm_type, axis=1)

def remove_empty_segments(times, labels):
    """Removes empty segments if needed."""
    assert len(times) - 1 == len(labels)
    inters = times_to_intervals(times)
    new_inters = []
    new_labels = []
    for inter, label in zip(inters, labels):
        if inter[0] < inter[1]:
            new_inters.append(inter)
            new_labels.append(label)
    return intervals_to_times(np.asarray(new_inters)), new_labels

def times_to_intervals(times):
    """Given a set of times, convert them into intervals.

    Parameters
    ----------
    times: np.array(N)
        A set of times.

    Returns
    -------
    inters: np.array(N-1, 2)
        A set of intervals.
    """
    return np.asarray(list(zip(times[:-1], times[1:])))

def intervals_to_times(inters):
    """Given a set of intervals, convert them into times.

    Parameters
    ----------
    inters: np.array(N-1, 2)
        A set of intervals.

    Returns
    -------
    times: np.array(N)
        A set of times.
    """
    return np.concatenate((inters.flatten()[::2], [inters[-1, -1]]), axis=0)