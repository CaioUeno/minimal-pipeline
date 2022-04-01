import numpy as np


class BaseClassifier:
    def __init__(self):
        pass

    def check_X_y(self, X, y):

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be of type np.ndarray but is {type(X)}")

        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be of type np.ndarray but is {type(y)}")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length: X ({len(X)}) and y ({len(y)})"
            )

        return True
