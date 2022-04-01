from statistics import mode
from scipy.special import softmax
from typing import Callable, Union

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.models.BaseClassifier import BaseClassifier


def euclidean(a: np.ndarray, b: np.ndarray):
    """Euclidean distance."""
    return np.linalg.norm(a - b)


def manhattan(a: np.ndarray, b: np.ndarray):
    """Manhattan distance."""
    return abs(a - b).sum()


class NearestNeighbors(BaseClassifier):

    """
    NearestNeighbors algorithm.

    Arguments:
        k
        metric
        n_jobs
    """

    def __init__(
        self,
        k: int = 3,
        metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
        n_jobs: int = 1,
    ) -> None:

        self.k = k
        self.metric = (
            metric
            if not isinstance(metric, str)
            else NearestNeighbors.pre_defined_metrics(metric)
        )
        self.n_jobs = n_jobs

    @staticmethod
    def pre_defined_metrics(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:

        """
        Return function given an known name (string).

        Args:
            metric (str): name of a known metric distance.

        Raises:
            NotImplementedError: if name is unknown.

        Returns:
            Callable[[np.ndarray, np.ndarray], float]: respective function for the given name.
        """

        if metric == "euclidean":
            return euclidean
        elif metric == "manhattan":
            return manhattan
        else:
            raise NotImplementedError(f"Unknown metric name.")

    def fit(self, X: np.ndarray, y: np.ndarray):

        """
        Fits the model using the provided training data (X, y).

        Arguments:
            X:;
            y:.

        Returns:
            itself.
        """

        # check whether input is valid or not
        self.check_X_y(X, y)

        # simply store the data
        self.X = X
        self.y = y

        # store metadata as well
        self.labels = np.unique(y)

        return self

    def predict(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:

        """
        Predict labels for the given set (X).

        Arguments:
            X: ;
            verbose: flag to indicate whether to show a progress bar or not.

        Returns:
            predictions: predicted class labels for each instance on X.
        """

        predictions = np.empty(len(X))

        for idx, x in enumerate(tqdm(X)) if verbose else enumerate(X):

            # calculate the distance for every training data
            dists = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                [delayed(self.metric)(x, i) for i in self.X]
            )

            # sort and retrieve the nearest neighbors indices
            neighbors_idx = np.argsort(dists)[: self.k]

            # retrieve their labels
            neighbors_labels = self.y[neighbors_idx]

            # calculate the mode
            predictions[idx] = mode(neighbors_labels)

        return predictions

    def predict_proba(self, X: np.ndarray, verbose: bool = True):

        """
        Predict class labels probabilities for the given set (X).

        Arguments:
            X: ;
            verbose: flag to indicate whether to show a progress bar or not.

        Returns:
            predictions: class labels probabilities for each instance on X.
        """

        predictions = np.empty((len(X), len(self.labels)))

        for idx, x in enumerate(tqdm(X)) if verbose else enumerate(X):

            # calculate the distance for every training data
            dists = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                [delayed(self.metric)(x, i) for i in self.X]
            )

            # sort and retrieve the nearest neighbors indices
            neighbors_idx = np.argsort(dists)[: self.k]

            # retrieve their labels
            neighbors_labels = self.y[neighbors_idx]

            # labels count
            labels_count = np.bincount(neighbors_labels, minlength=len(self.labels))

            # softmax to estimate a probability distribution
            predictions[idx, :] = softmax(labels_count)

        return predictions
