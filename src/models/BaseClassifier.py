import os
from abc import abstractmethod
from shutil import make_archive
from zipfile import ZipFile

import numpy as np


class NotMatchingLengthError(Exception):
    """Raised when X and y have different lengths."""

    pass


class FitError(Exception):
    """Raised when classifier was not fitted (some actions require it)."""

    pass


class BaseClassifier:

    """
    Base class to share common methods.
    """

    def __init__(self):
        pass

    def check_X_y(self, X, y) -> bool:

        """
        Check whether inputs are valid.

        Arguments:
            X: expected feature matrix;
            y: expected labels.

        Raises:
            TypeError: if one of the inputs is not of type np.ndarray;
            NotMatchingLengthError: if X and y have different lengths.

        Returns:
            True
        """

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be of type np.ndarray but is {type(X)}")

        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be of type np.ndarray but is {type(y)}")

        if len(X) != len(y):
            raise NotMatchingLengthError(
                f"X and y must have same length: X ({len(X)}) and y ({len(y)})"
            )

        return True

    def save(self, path: str) -> bool:

        """
        Common method to check if is possible to save the model, and to create the directory.

        Arguments:
            path (str): directory's path to store the model's parameters.

        Raises:
            FitError: if classifier was not fitted yet (no parameters to save).

        Returns:
            True
        """

        if not self.fitted:
            raise FitError(f"Classifier must be fitted before it can be saved.")

        if not os.path.isdir(path):
            os.mkdir(path)

        return True

    def compress(self, path: str) -> str:

        """
        Compress the directory into a zip file.


        Arguments:
            path (str): directory's path where the model's parameters are stored.

        Returns:
            compressed_file_path (str): compressed file path.
        """

        compressed_file_path = make_archive(path, "zip", path)

        return compressed_file_path

    def uncompress(self, path: str) -> str:

        """
        Uncompress a file zip containing a model's parameters.

        Arguments:
            path (str): zip file path.

        Returns:
            uncompressed_path (str): extracted directory's path.
        """

        if ".zip" not in path:
            raise ValueError(f"Path argument must be a zip file: {path}")

        uncompressed_path = path.split(".zip")[0]

        with ZipFile(path, "r") as zip:
            zip.extractall(uncompressed_path)

        return uncompressed_path

    @abstractmethod
    def load(self, path: str):

        """
        Abtract method to load a model from directory.

        Arguments:
            path (str): directory's path where the model's parameters is store.
        """

        pass
