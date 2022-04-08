import numpy as np

from src.models.BaseClassifier import BaseClassifier


class Node:

    """Base class to share common methods."""

    def __init__(self):
        pass


class LeafNode(Node):

    """
    Class to represent a leaf node.

    Arguments:
        label (int): label to return if instance reaches it.
    """

    def __init__(self, label: int):
        self.label = label

    def predict(self, x: np.ndarray) -> int:

        """
        Predict a single instance.

        Arguments:
            x (np.ndarray): input instance.

        Returns:
            int: predicted label.
        """

        return self.label


class InternalNode(Node):

    """
    Class to represent an internal node.

    Arguments:
        feature: index of the feature to base decision;
        threshold: split point value.
    """

    def __init__(self, feature: int, threshold: float):
        self.feature = feature
        self.threshold = threshold

    def predict(self, x: np.ndarray) -> int:

        """
        Predict a single instance.

        Args:
            x (np.ndarray): input instance.

        Returns:
            int: predicted label.
        """

        if x[self.feature] < self.threshold:
            return self.left.predict(x)

        return self.right.predict(x)

    def set_left(self, node: Node):
        """Set the left child."""
        self.left = node

    def set_right(self, node: Node):
        """Set the right child."""
        self.right = node


class DecisionTree(BaseClassifier):

    """
    Decision Tree classifier.

    Arguments:
        max_depth (int, optional): _description_. Defaults to None.
        random_state (int, optional): _description_. Defaults to 0.
    """

    def __init__(self, max_depth: int = None, random_state: int = 0):
        self.max_depth = max_depth
        self.random_state = random_state

    @staticmethod
    def entropy(y: np.ndarray) -> float:

        """
        Calculate the entropy of classes of given samples.

        Arguments:
            y: samples' labels.

        Returns:
            entropy: entropy value.
        """

        _, counts = np.unique(y, return_counts=True)
        p = counts / len(counts)
        entropy = -(p @ np.log(p).T)

        return entropy

    def build_node(self, X: np.ndarray, y: np.ndarray, features: np.ndarray):
        """_summary_

        Args:
            X (np.ndarray): _description_
            y (np.ndarray): _description_
            features (np.ndarray): _description_

        Returns:
            _type_: _description_
        """


        # no data
        if len(X) == 0:
            return None

        # only one class
        if len(np.unique(y)) == 1:
            return LeafNode(label=np.unique(y)[0])

        # two or more classes
        else:

            # no more features available - return leaf node using most frequent label
            if len(features) == 0:
                return LeafNode(label=np.argmax(np.bincount(y)))

            best_splitpoints = {}

            # iterate over features and thresholds
            for feature in features:

                min_value, max_value = min(X[:, feature]), max(X[:, feature])
                n_splitpoints = len(X) + 1
                splitpoints = np.linspace(min_value, max_value, n_splitpoints)
                information_gain = np.empty(n_splitpoints)

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

            # check if there is data to build children
            if (X[:, feature] < splitpoint).sum() > 0:

                left_child = self.build_node(
                    X[X[:, feature] < splitpoint],
                    y[X[:, feature] < splitpoint],
                    list(
                        filter(lambda f: f != feature, features)
                    ),  # remove selected feature from available list
                )
                node.set_left(left_child)

            if (X[:, feature] >= splitpoint).sum() > 0:
                right_child = self.build_node(
                    X[X[:, feature] >= splitpoint],
                    y[X[:, feature] >= splitpoint],
                    list(
                        filter(lambda f: f != feature, features)
                    ),  # remove selected feature from available list
                )
                node.set_right(right_child)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray):

        """
        Fit the model using the provided training data (X, y).
        Create a tree to represent the decisions.

        Arguments:
            X: matrix of training instances' features of shape (n_instances, n_features);
            y: instances' labels of shape (n_instances).

        Returns:
            itself.
        """

        self.tree = self.build_node(X, y, list(range(X.shape[1])))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        """
        Predict labels for the given set (X).

        Arguments:
            X: matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions: predicted class labels for each instance on X.
        """

        predictions = [self.tree.predict(x) for x in X]

        return predictions
