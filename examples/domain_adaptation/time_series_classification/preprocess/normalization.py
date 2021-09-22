"""
Calculate/apply normalization
"""
import numpy as np


def calc_normalization(x, method):
    """
    Calculate zero mean unit variance normalization statistics

    We calculate separate mean/std or min/max statistics for each
    feature/channel, the default (-1) for BatchNormalization in TensorFlow and
    I think makes the most sense. If we set axis=0, then we end up with a
    separate statistic for each time step and feature, and then we can get odd
    jumps between time steps. Though, we get shape problems when setting axis=2
    in numpy, so instead we reshape/transpose.
    """
    # from (10000,100,1) to (1,100,10000)
    x = x.T
    # from (1,100,10000) to (1,100*10000)
    x = x.reshape((x.shape[0], -1))
    # then we compute statistics over axis=1, i.e. along 100*10000 and end up
    # with 1 statistic per channel (in this example only one)

    if method == "meanstd":
        values = (np.mean(x, axis=1), np.std(x, axis=1))
    elif method == "minmax":
        values = (np.min(x, axis=1), np.max(x, axis=1))
    else:
        raise NotImplementedError("unsupported normalization method")

    return method, values


def is_numpy(x):
    # Though, could probably use isinstance(x, np.ndarray) ?
    # https://stackoverflow.com/a/12570040
    return type(x).__module__ == np.__name__


def to_numpy_if_not(x, dtype=np.float32):
    # Create a numpy array, if not one already
    if not is_numpy(x):
        x = np.array(x, dtype=dtype)

    return x


def calc_normalization_jagged(x, method):
    """ Same as calc_normalization() except works for arrays of varying-length
    numpy arrays

    x should be: [
        np.array([example 1 time steps, example 1 features]),
        np.array([example 2 time steps, example 2 features]),
        ...
    ] where the # time steps can differ between examples.
    """
    assert len(x) > 0, "x cannot be zero-length"

    # No data, e.g. GPS sometimes has no values in the window, so find a window
    # that does have data to get shape -- otherwise we can't normalize since
    # there is no data, so skip by returning (None, None)
    found = False
    for example in x:
        if example.shape != (0,):
            assert len(example.shape) > 1, \
                "shape should be [time steps, features] but is " \
                + str(example.shape)
            num_features = example.shape[1]
            found = True
            break

    if not found:
        print("Warning: no data found, so skipping normalization")
        return (None, None)

    # if is_numpy(x[0]):
    # assert len(x[0].shape) > 1, "shape should be [time steps, features] but is " \
    #     + str(x[0].shape)
    # # Get feature dimension from example zero
    # num_features = x[0].shape[1]
    # else:
    #     found = False
    #     for example in x:
    #         if len(example) > 0:
    #             # Get the number of dimensions from this example's
    #             # time dimension's number of features
    #             num_features = len(example[0])
    #             found = True
    #             break
    #     assert found, "need at least one example to have at least one time step"

    features = [None for x in range(num_features)]

    for example in x:
        # example = to_numpy_if_not(example)
        transpose = example.T

        for i, feature_values in enumerate(transpose):
            if features[i] is None:
                features[i] = feature_values
            else:
                features[i] = np.concatenate([features[i], feature_values], axis=0)

    if method == "meanstd":
        values = (np.array([np.mean(x) for x in features], dtype=np.float32),
            np.array([np.std(x) for x in features], dtype=np.float32))
    elif method == "minmax":
        values = (np.array([np.min(x) for x in features], dtype=np.float32),
            np.array([np.max(x) for x in features], dtype=np.float32))
    else:
        raise NotImplementedError("unsupported normalization method")

    return method, values


def apply_normalization(x, normalization, epsilon=1e-5):
    """ Apply zero mean unit variance normalization statistics """
    # Don't do anything if it's zero-length
    if len(x) == 0:
        return x

    method, values = normalization

    if method == "meanstd":
        mean, std = values
        x = (x - mean) / (std + epsilon)
    elif method == "minmax":
        minx, maxx = values
        x = (x - minx) / (maxx - minx + epsilon) - 0.5

    x[np.isnan(x)] = 0

    return x


def apply_normalization_jagged(x, normalization, epsilon=1e-5):
    """ Same as apply_normalization() except works for arrays of varying-length
    numpy arrays """
    # Can't normalize if there was no normalization statistics computed
    if normalization[0] is None or normalization[1] is None:
        return x

    normalized = []

    for example in x:
        normalized.append(apply_normalization(example, normalization, epsilon))

    return normalized
