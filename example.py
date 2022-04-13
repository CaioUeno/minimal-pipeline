import numpy as np
from sklearn.metrics import accuracy_score

from src.models.DecisionTree import DecisionTree
from src.models.LogisticRegression import LogisticRegression
from src.models.NaiveBayes import GaussianNaiveBayes
from src.models.NearestNeighbors import NearestNeighbors


def main():

    # select a model
    # model = NearestNeighbors(k=3, metric="euclidean", n_jobs=-1)
    # model = GaussianNaiveBayes(priors=None)
    model = LogisticRegression()
    # model = DecisionTree(max_depth=5)

    # training data
    X_train = np.array([[0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]])
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model.fit(X_train, y_train)

    # test data
    X_test = np.array([[0.15], [0.25], [0.35], [0.45], [0.55]])
    y_test = np.array([0, 0, 1, 0, 1])

    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))}")

    model.save("model")


if __name__ == "__main__":
    main()
