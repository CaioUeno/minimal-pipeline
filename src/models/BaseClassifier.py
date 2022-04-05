from abc import abstractmethod
import os
from shutil import make_archive
from zipfile import ZipFile

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

        if not self.fitted:
            raise ValueError(f"Classifier must be fitted before it can be saved.")

        if not os.path.isdir(path):
            os.mkdir(path)
            
        return True

    @abstractmethod
    def compress(self, path: str) -> str:
        return make_archive(path, "zip", path)

    @abstractmethod
    def uncompress(self, path: str) -> str:

        if ".zip" not in path:
            return ""

        uncompressed_path = path.split(".zip")[0]

        with ZipFile(path, "r") as zip:
            zip.extractall(uncompressed_path)

        return uncompressed_path

    @abstractmethod
    def load(self, path: str):
        pass
