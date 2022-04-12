import numpy as np
from scipy.special import softmax
from src.models.BaseClassifier import BaseClassifier
from tqdm import tqdm


class LogisticRegression(BaseClassifier):

    """Logistic Regression algorithm."""

    def __init__(
        self,
    ):
        self.fitted = False

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:

        """
        Calculate the loss function.

        Arguments:
            y_true (np.ndarray): expected output;
            y_pred (np.ndarray): model's predicted output.

        Returns:
            loss (float): loss value.
        """

        loss = -(y_true * np.log(y_pred)).sum(axis=1)

        return loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 16,
        epochs: int = 100,
        learning_rate: float = 0.1,
        shuffle: bool = True,
        verbose: bool = False,
    ):

        """
        Fit model using back propagation.

        Arguments:
            X (np.ndarray): matrix of training instances' features of shape (n_instances, n_features);
            y (np.ndarray): instances' labels of shape (n_instances);
            batch_size (int, optional): number of instances in a batch. Defaults to 32.
            epochs (int, optional): number of epochs to iterate. Defaults to 100.
            learning_rate (float, optional): pace to control update. Defaults to 0.1.
            shuffle (bool, optional): whether to shuffle data before training. Defaults to False.
            verbose (bool, optional): flag to indicate whether to show a progress bar or not. Defaults to False.

        Returns:
            itself.
        """

        # check whether input is valid or not
        self.check_X_y(X, y)

        # shuffle input
        if shuffle:
            indices = np.random.permutation(len(X))
        else:
            indices = np.array(range(len(X)))

        # use new vars to avoid changing the original data (X and y)
        X_train, y_train = X[indices].copy(), y[indices].copy()

        # transform into one hot encoding
        y_train = np.stack(
            [np.bincount([label], minlength=max(y) + 1) for label in y_train]
        )

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        # initialize weights and biases - normal distribution between [-1, 1]
        self.weights = 2 * np.random.random((self.n_classes, X.shape[1])) - 1
        self.bias = 2 * np.random.random((self.n_classes, 1)) - 1

        # iterate over epochs
        for _ in tqdm(range(epochs), unit="epoch") if verbose else range(epochs):

            # iterate over batches
            for i in range(0, len(X_train), batch_size):

                # get instances for batch i
                x_batch = X_train[i : i + batch_size, :]
                y_batch = y_train[i : i + batch_size, :]

                # feed forward
                lr = self.weights @ x_batch.T + self.bias

                # weights derivative
                wdelta = (softmax(lr, axis=0) - y_batch.T) @ x_batch

                # add learning rate and smooth by batch_size
                wdelta = learning_rate * wdelta / batch_size

                # update
                self.weights -= wdelta

        self.fitted = True

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        """
        Predict class labels probabilities for the given set (X).

        Arguments:
            X (np.ndarray): matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions (np.ndarray): class labels probabilities for each instance on X.
        """

        predictions = softmax(self.weights @ X.T + self.bias, axis=1).T

        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:

        """
        Predict labels for the given set (X).

        Arguments:
            X (np.ndarray): matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions (np.ndarray): predicted class labels for each instance on X.
        """

        predictions = self.predict_proba(X).argmax(axis=1)

        return predictions

    def save(self, path: str) -> bool:

        """
        Save model's parameters.

        Arguments:
            path (str): directory's path to save the model.

        Returns:
            bool: whether the model was saved successfully or not.
        """

        super().save(path)

        with open(f"{path}/classes.npy", "wb") as f:
            np.save(f, self.classes)

        with open(f"{path}/weights.npy", "wb") as f:
            np.save(f, self.weights)

        with open(f"{path}/bias.npy", "wb") as f:
            np.save(f, self.bias)

        return True

    def load(self, path: str):

        """
        Load model from directory path.

        Arguments:
            path (str): directory's path where the model's parameters are.

        Returns:
            itself.
        """

        with open(f"{path}/classes.npy", "rb") as f:
            self.classes = np.load(f)

        self.n_classes = len(self.classes)

        with open(f"{path}/weights.npy", "rb") as f:
            self.weights = np.load(f)

        with open(f"{path}/bias.npy", "rb") as f:
            self.bias = np.load(f)

        return self
