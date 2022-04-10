import numpy as np
from scipy.special import softmax
from src.models.BaseClassifier import BaseClassifier
from tqdm import tqdm


class LogisticRegression(BaseClassifier):

    """Logistic Regression algorithm."""

    def __init__(
        self,
    ):
        pass

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray):

        return -(y_true * np.log(y_pred)).sum(axis=1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.01,
        shuffle: bool = True,
        verbose: bool = True,
    ):

        """
        Fit model

        Arguments:
            X: matrix of training instances' features of shape (n_instances, n_features);
            y: instances' labels as one-hot encoding of shape (n_instances);
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

        # use new vars to avoid changing the original data (X and y)
        if shuffle:
            p = np.random.permutation(len(X))
            X_train, y_train = X[p].copy(), y[p].copy()

        else:
            X_train, y_train = X.copy(), y.copy()

        y_train = np.stack([np.bincount([label], minlength=max(y) + 1) for label in y])
        self.classes = np.unique(y)

        # initialize weights and biases - normal distribution between [-1, 1]
        self.weights = 2 * np.random.random((X.shape[1], len(self.classes))) - 1
        self.bias = np.random.random(len(self.classes)) - 0.5

        # iterate over epochs
        for epoch in tqdm(range(epochs)) if verbose else range(epochs):

            # iterate over batches
            for i in range(0, len(X_train), batch_size):

                # feed forward
                lr = X_train[i : i + batch_size, :] @ self.weights + self.bias

                # derivative
                delta = X_train[i : i + batch_size, :].T @ (
                    softmax(lr) - y_train[i : i + batch_size, :]
                )

                # add learning rate and smooth by batch_size
                delta = learning_rate * delta / batch_size

                print(self.weights)
                self.weights -= delta
                print(self.weights)
                print(delta)

                print(
                    LogisticRegression.loss(
                        y_train[i : i + batch_size, :], softmax(lr)
                    ).mean()
                )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        """
        Predict class labels probabilities for the given set (X).

        Arguments:
            X: matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions: class labels probabilities for each instance on X.
        """

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
