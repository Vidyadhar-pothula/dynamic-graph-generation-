import numpy as np

def fuse_features(appearance, spatial, motion):
    """
    Fuses appearance (512,), spatial (9,), and motion (6,) features into a single vector.
    Returns:
        numpy array of shape (527,)
    """
    return np.concatenate([appearance, spatial, motion], axis=-1)
