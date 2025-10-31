import cupy as np


def kernel(r):

    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta # set alpha to (-1/(beta)) * sum ( in[k], cdot(flipped in up to k, out up to k)). Use reverse indexing and blocking.
        y[:k] += alpha * np.flip(y[:k]) # flip first k values of output and multiply by alpha. Could be done without flipping using reverse indexing + smart blocking.
        y[k] = alpha # can be parallelized with the above operation

    return y
