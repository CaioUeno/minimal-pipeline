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
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):

        """
        Fit the model using the provided training data (X, y).
        Estimate priors if not provided and for each class calculate mean and variance of features.

        Arguments:
            X: matrix of training instances' features of shape (n_instances, n_features);
            y: instances' labels of shape (n_instances).

        Returns:
            itself.
        """

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

        self.fitted = True

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        """
        Predict class labels probabilities for the given set (X).

        Arguments:
            X: matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions: class labels probabilities for each instance on X.
        """

        likelihood = np.zeros((X.shape[0], len(self.classes)), dtype=float)

        for c in self.classes:

            lc = np.exp(-0.5 * (X - self.means[c, :]) ** 2 / (self.vars[c, :] ** 2))
            lc *= 1 / (self.vars[c, :] * np.sqrt(2 * np.pi))
            likelihood[:, c] = lc.prod(axis=1)

        # since P(x) is not calculated, use softmax to
        # ensure a probability distribution (sum up to 1)
        predictions = softmax(likelihood * self.priors, axis=1)

        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:

        """
        Predict labels for the given set (X).

        Arguments:
            X: matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions: predicted class labels for each instance on X.
        """

        predictions = self.predict_proba(X).argmax(axis=1)

        return predictions

    def save(self, path: str) -> bool:

        super().save(path)

        with open(f"{path}/priors.npy", "wb") as f:
            np.save(f, self.priors)

        with open(f"{path}/classes.npy", "wb") as f:
            np.save(f, self.classes)

        with open(f"{path}/means.npy", "wb") as f:
            np.save(f, self.means)

        with open(f"{path}/vars.npy", "wb") as f:
            np.save(f, self.vars)

        return True

    def load(self, path: str):

        with open(f"{path}/priors.npy", "rb") as f:
            self.priors = np.load(f)

        with open(f"{path}/classes.npy", "rb") as f:
            self.classes = np.load(f)

        with open(f"{path}/means.npy", "rb") as f:
            self.means = np.load(f)

        with open(f"{path}/vars.npy", "rb") as f:
            self.vars = np.load(f)

        return self
