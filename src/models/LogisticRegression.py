import numpy as np
from scipy.special import softmax
from src.models.BaseClassifier import BaseClassifier


class LogisticRegression(BaseClassifier):

    """Logistic Regression algorithm."""

    def __init__(
        self,
    ):
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 2,
        epochs: int = 10,
        learning_rate: float = 0.01,
        shuffle: bool = True,
        verbose: bool = True,
    ):

        """
        Fit model

        Arguments:
            X: matrix of training instances' features of shape (n_instances, n_features);
            y: instances' labels as one-hot encoding of shape (n_instances, n_classes);
            batch_size: number of instances in a batch;
            epochs: number of epochs to iterate;
            learning_rate: pace to control update rate;
            shuffle: whether to shuffle data before training;
            verbose: flag to indicate whether to show a progress bar or not.

        Returns:
            training_loss: ;
            evaluation_loss: ;

        """

        # check whether input is valid or not
        self.check_X_y(X, y)

        self.classes = np.unique(y)

        # initialize weights and biases
        self.weights = 2 * np.random.random((X.shape[1], len(self.classes))) - 1
        self.bias = 2 * np.random.random(len(self.classes)) - 1

        # iterate over epochs
        for epoch in range(epochs):

            for i in range(0, len(X), batch_size):

                lr = X[i : i + batch_size, :] @ self.weights + self.bias
                delta = X[i : i + batch_size, :].T @ (lr - y[i : i + batch_size, :])
                delta = learning_rate * delta * (1 / batch_size)

                self.weights -= delta

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        """
        Predict class labels probabilities for the given set (X).

        Arguments:
            X: matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions: class labels probabilities for each instance on X.
        """

        # apply an softmax funtion to ensure a probability distribution
        predictions = softmax(X @ self.weights + self.bias, axis=1)

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
