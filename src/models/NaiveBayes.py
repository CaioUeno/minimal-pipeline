import numpy as np
from scipy.special import softmax
from src.models.BaseClassifier import BaseClassifier


class GaussianNaiveBayes(BaseClassifier):

    """
    Naive Bayes algorithm.

    Arguments:
        priors: classes' prior probabilities.
    """

    def __init__(self, priors: np.ndarray = None):
        self.priors = priors

    def fit(self, X: np.ndarray, y: np.ndarray):

        # """
        # Fit the model using the provided training data (X, y).
        # Calculate

        # Arguments:
        #     X: matrix of training instances' features of shape (n_instances, n_features);
        #     y: instances' labels of shape (n_instances).

        # Returns:
        #     itself.
        # """

        # check whether input is valid or not
        self.check_X_y(X, y)

        # use class frequency as classes' prior probabilities
        # if it hasn't been passed
        if self.priors is None:
            self.priors = np.bincount(y) / np.bincount(y).sum()

        self.classes = np.unique(y)

        self.means = np.zeros((len(self.classes), X.shape[1]), dtype=float)
        self.vars = np.zeros((len(self.classes), X.shape[1]), dtype=float)

        # for each class calculate mean and variance of features
        for c in self.classes:
            self.means[c, :] = X[y == c, :].mean()
            self.vars[c, :] = X[y == c, :].std()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        """_summary_

        Args:
            X: matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            np.ndarray: _description_
        """

        likelihood = np.zeros((X.shape[0], len(self.classes)), dtype=float)

        for c in self.classes:
            print(f"class {c}")
            lc = np.exp(-0.5 * (X - self.means[c, :]) ** 2 / (self.vars[c, :] ** 2))
            lc *= 1 / (self.vars[c, :] * np.sqrt(2 * np.pi))
            likelihood[:, c] = lc.prod(axis=1)
            print("--------")

        proba = likelihood * self.priors

        return softmax(proba, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:

        """_summary_

        Args:
            X: matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            np.ndarray: _description_
        """

        return self.predict_proba(X).argmax(axis=1)
