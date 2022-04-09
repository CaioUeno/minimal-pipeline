from typing import List
import numpy as np

from src.models.BaseClassifier import BaseClassifier


class NoDataError(Exception):
    """Raised when the input data is empty."""

    pass


class Node:

    """Base Node class."""

    def __init__(self):
        pass


class LeafNode(Node):

    """
    Class to represent a leaf node.

    Arguments:
        label (int): label to return if instance reaches it.
    """

    def __init__(self, label: int, class_proportions: np.ndarray):
        self.label = label
        self.class_proportions = class_proportions

    def predict(self, x: np.ndarray) -> int:

        """
        Predict a single instance.

        Arguments:
            x (np.ndarray): input instance.

        Returns:
            int: predicted label.
        """

        return self.label

    def predict_proba(self, x: np.ndarray) -> np.ndarray:

        """
        Predict class probabilities of a single instance.

        Args:
            x (np.ndarray):  input instance.

        Returns:
            np.ndarray: class proportions as probability distribution.
        """

        return self.class_proportions


class InternalNode(Node):

    """
    Class to represent an internal node.

    Arguments:
        feature (int): index of the feature to base decision;
        threshold (float): split point value.
    """

    def __init__(self, feature: int, threshold: float):
        self.feature = feature
        self.threshold = threshold

    def set_left(self, node: Node):
        """Set the left child."""
        self.left = node

    def set_right(self, node: Node):
        """Set the right child."""
        self.right = node

    def predict(self, x: np.ndarray) -> int:

        """
        Predict a single instance.

        Arguments:
            x (np.ndarray): input instance.

        Returns:
            int: predicted label.
        """

        if x[self.feature] < self.threshold:
            return self.left.predict(x)

        return self.right.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:

        """
        Predict class probabilities of a single instance.

        Args:
            x (np.ndarray):  input instance.

        Returns:
            np.ndarray: class proportions as probability distribution.
        """

        if x[self.feature] < self.threshold:
            return self.left.predict_proba(x)

        return self.right.predict_proba(x)


class DecisionTree(BaseClassifier):

    """
    Decision Tree classifier.

    Arguments:
        max_depth (int, optional): maximum depth (heigth) of the tree. Defaults to None - unlimited.
    """

    def __init__(self, max_depth: int = None):
        self.max_depth = max_depth
        self.fitted = False

    @staticmethod
    def entropy(y: np.ndarray) -> float:

        """
        Calculate the entropy of classes of given samples.
        Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)

        Arguments:
            y (np.ndarray): samples' labels.

        Returns:
            entropy (float): entropy value.
        """

        _, counts = np.unique(y, return_counts=True)
        p = counts / len(counts)
        entropy = -(p @ np.log(p).T)

        return entropy

    def build_node(
        self, X: np.ndarray, y: np.ndarray, features: List[int], depth: int
    ) -> Node:

        """
        Create a node given a set of samples and available features.

        Arguments:
            X (np.ndarray): matrix of training instances' features of shape (n_instances, n_features);
            y (np.ndarray): instances' labels of shape (n_instances);
            features (List[int]): list of available features (index).

        Returns:
            node (Node): the create node. It can be a LeafNode or an InternalNode.
        """

        # no data
        if len(X) == 0:
            raise NoDataError(f"No samples to build a node.")

        # only one class
        if len(np.unique(y)) == 1:

            unique_label = np.unique(y)[0]
            proportions = np.bincount(y, minlength=len(self.classes))
            return LeafNode(
                label=unique_label,
                class_proportions=proportions / proportions.sum(),
            )

        # two or more classes
        else:

            # no more features available or max_depth - return leaf node using most frequent label
            if len(features) == 0 or depth == self.max_depth:

                most_frequent_label = np.argmax(np.bincount(y))
                proportions = np.bincount(y, minlength=len(self.classes))

                return LeafNode(
                    label=most_frequent_label,
                    class_proportions=proportions / proportions.sum(),
                )

            best_splitpoints = {}

            # iterate over features and thresholds
            for feature in features:

                # create array with possible split points
                min_value, max_value = min(X[:, feature]), max(X[:, feature])
                splitpoints = np.linspace(min_value, max_value, len(X) + 1)[1:-1]

                information_gain = np.empty(len(splitpoints))

                # iterate over possible split points
                for idx, splitpoint in enumerate(splitpoints):

                    # calculate information gain
                    information_gain[idx] = DecisionTree.entropy(y) - sum(
                        [
                            DecisionTree.entropy(y[X[:, feature] < splitpoint]),
                            DecisionTree.entropy(y[X[:, feature] >= splitpoint]),
                        ]
                    )

                # splitpoint that maximizes information gain
                best_splitpoints[feature] = splitpoints[np.argmax(information_gain)]

            feature = max(best_splitpoints, key=best_splitpoints.get)
            splitpoint = best_splitpoints[feature]

            # create respective node
            node = InternalNode(feature=feature, threshold=splitpoint)

            # remove selected feature from available list
            update_available_features = list(filter(lambda f: f != feature, features))

            # set children
            node.set_left(
                self.build_node(
                    X[X[:, feature] < splitpoint],
                    y[X[:, feature] < splitpoint],
                    update_available_features,
                    depth + 1,
                )
            )

            node.set_right(
                self.build_node(
                    X[X[:, feature] >= splitpoint],
                    y[X[:, feature] >= splitpoint],
                    update_available_features,
                    depth + 1,
                )
            )

        return node

    def fit(self, X: np.ndarray, y: np.ndarray):

        """
        Fit the model using the provided training data (X, y).
        Create a tree to represent the decisions.

        Arguments:
            X (np.ndarray): matrix of training instances' features of shape (n_instances, n_features);
            y (np.ndarray): instances' labels of shape (n_instances).

        Returns:
            itself.
        """

        self.classes = np.unique(y)

        self.tree = self.build_node(X, y, list(range(X.shape[1])), 0)

        self.fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        """
        Predict labels for the given set (X).

        Arguments:
            X (np.ndarray): matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions (np.ndarray): predicted class labels for each instance on X.
        """

        predictions = np.array([self.tree.predict(x) for x in X])

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        """
        Predict class labels probabilities for the given set (X).

        Arguments:
            X: matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions: class labels probabilities for each instance on X.
        """

        predictions = np.stack([self.tree.predict_proba(x) for x in X])

        return predictions

    def save(self, path: str) -> bool:

        super().save(path)

        # with open(f"{path}/X.npy", "wb") as f:
        #     np.save(f, self.X)

        # with open(f"{path}/y.npy", "wb") as f:
        #     np.save(f, self.y)

        # with open(f"{path}/labels.npy", "wb") as f:
        #     np.save(f, self.labels)

        return True

    def load(self, path: str):

        # with open(f"{path}/X.npy", "rb") as f:
        #     self.X = np.load(f)

        # with open(f"{path}/y.npy", "rb") as f:
        #     self.y = np.load(f)

        # with open(f"{path}/labels.npy", "rb") as f:
        #     self.labels = np.load(f)

        return self
