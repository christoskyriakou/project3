import numpy as np
import struct
import os

def L2_distance_batch(A, B):
    """
    Compute L2 distances between each row of A and vector B.
    A: (n, d)
    B: (d,)
    Returns: (n,) array of distances
    """
    diff = A - B
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    return dists