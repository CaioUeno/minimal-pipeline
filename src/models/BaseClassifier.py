from abc import abstractmethod
import numpy as np


class BaseClassifier:

    """
    Base class to share common methods.
    """

    def __init__(self):
        pass

    def check_X_y(self, X, y) -> bool:

        """
        Check whether inputs are valid.

        Raises:
            TypeError: if one of the inputs is not of type np.ndarray;
            ValueError: _description_

        Returns:
            True
        """

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be of type np.ndarray but is {type(X)}")

        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be of type np.ndarray but is {type(y)}")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length: X ({len(X)}) and y ({len(y)})"
            )

        return True

    @abstractmethod
    def save(self, path: str) -> bool:
        pass

    @abstractmethod
    def load(self, path: str):
        pass