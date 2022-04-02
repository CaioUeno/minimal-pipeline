import numpy as np
from scipy.special import softmax
from src.models.BaseClassifier import BaseClassifier


class LogisticRegression(BaseClassifier):
    def __init__(
        self,
    ):

        pass

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 2):

        # check whether input is valid or not
        self.check_X_y(X, y)

        self.classes = np.unique(y)

        self.weights = 2 * np.random.random((X.shape[1], len(self.classes))) - 1
        self.bias = 2 * np.random.random(len(self.classes)) - 1

        for _ in range(25):
            lr = X @ self.weights + self.bias

            d = (lr - y) * 1
            # act = softmax(lr, axis=1)

            # d = act - y
            # # print("weights")
            # # print(self.weights)
            # # print("delta")
            # # print(0.001 * (X.T @ d) * (1 / len(X)))

            # error = (-y * np.log(act)).mean(axis=0).sum()
            error = (lr - y).mean(axis=0) 
            print(error)
            self.weights -= 0.01 * (X.T @ d) * (1 / len(X))

        return self

    def forward(self, X: np.ndarray):
        return softmax(X @ self.weights + self.bias, axis=1)
