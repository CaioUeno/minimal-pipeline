import json
from typing import List
from uuid import uuid4
import numpy as np

from src.models.BaseClassifier import BaseClassifier


class NoDataError(Exception):
    """Raised when the input data is empty."""

    pass


class NodeNotFoundError(Exception):
    """Raised when node is not found."""

    pass


class Node:

    """Base Node class."""

    def __init__(self):
        pass


class LeafNode(Node):

    """
    Class to represent a leaf node.

    Arguments:
        label (int): label to return if instance reaches it;
        class_proportions (np.ndarray): class proportions of instances used to build this node.
    """

    def __init__(self, label: int, class_proportions: np.ndarray):
        self.id = str(uuid4())  # unique id to persist model easily
        self.label = label
        self.class_proportions = class_proportions

    def predict(self, x: np.ndarray) -> int:

        """
        Predict a single instance.

        Arguments:
            x (np.ndarray): input instance.

        Returns:
            prediction (int): predicted label.
        """

        prediction = self.label

        return prediction

    def predict_proba(self, x: np.ndarray) -> np.ndarray:

        """
        Predict class probabilities of a single instance.

        Arguments:
            x (np.ndarray): input instance.

        Returns:
            prediction (np.ndarray): class proportions as probability distribution.
        """

        prediction = self.class_proportions

        return prediction

    def info(self) -> dict:

        """
        Return node information.

        Returns:
            dict: key:value mapping of attributes.
        """

        return {
            "id": self.id,
            "label": str(self.label),
            "class_proportions": list(self.class_proportions),
        }

    def to_list(self, nodes: List[dict]) -> List[dict]:

        """
        Append node to given list.

        Arguments:
            nodes (List[dict]): list of nodes.

        Returns:
            nodes (List[dict]): updated list of nodes.
        """

        nodes.append(self.info())

        return nodes


class InternalNode(Node):

    """
    Class to represent an internal node.

    Arguments:
        feature (int): index of the feature to base decision;
        threshold (float): split point value.
    """

    def __init__(self, feature: int, threshold: float):
        self.id = str(uuid4())  # unique id to persist model easily
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
            prediction (int): predicted label.
        """

        if x[self.feature] < self.threshold:
            prediction = self.left.predict(x)
        else:
            prediction = self.right.predict(x)

        return prediction

    def predict_proba(self, x: np.ndarray) -> np.ndarray:

        """
        Predict class probabilities of a single instance.

        Arguments:
            x (np.ndarray):  input instance.

        Returns:
            prediction (np.ndarray): class proportions as probability distribution.
        """

        if x[self.feature] < self.threshold:
            prediction = self.left.predict_proba(x)
        else:
            prediction = self.right.predict_proba(x)

        return prediction

    def info(self) -> dict:

        """
        Return node information.

        Returns:
            dict: key:value mapping of attributes.
        """

        return {
            "id": self.id,
            "feature": self.feature,
            "threshold": self.threshold,
            "left": None if self.left is None else self.left.id,
            "right": None if self.right is None else self.right.id,
        }

    def to_list(self, nodes: List[dict]) -> List[dict]:

        """
        Append node (and its children) to given list.

        Arguments:
            nodes (List[dict]): list of nodes.

        Returns:
            nodes (List[dict]): updated list of nodes.
        """

        nodes.append(self.info())

        # append children as well
        if self.left is not None:
            nodes = self.left.to_list(nodes)

        if self.right is not None:
            nodes = self.right.to_list(nodes)

        return nodes


class DecisionTree(BaseClassifier):

    """
    Decision Tree classifier.

    Arguments:
        max_depth (int, optional): maximum depth (heigth) of the tree. Defaults to None - "unlimited".
    """

    def __init__(self, max_depth: int = None):
        self.max_depth = max_depth
        self.fitted = False
        self.n_nodes = 0

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
            features (List[int]): list of available features (index);
            depth (int): depth of the current node.

        Raises:
            NoDataError: No data was provided.

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

            self.n_nodes += 1

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

                self.n_nodes += 1

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

            self.n_nodes += 1

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

        self.tree = self.build_node(X, y, features=list(range(X.shape[1])), depth=0)

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
            X (np.ndarray): matrix of instances' features to evaluate of shape (n_instances, n_features).

        Returns:
            predictions (np.ndarray): class labels probabilities for each instance on X.
        """

        predictions = np.stack([self.tree.predict_proba(x) for x in X])

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

        # create a list containing all nodes as dictionaries
        nodes = self.tree.to_list([])

        # save as json file
        with open(f"{path}/tree.json", "w") as f:
            f.write(json.dumps(nodes))

        return True

    @staticmethod
    def search(idd: str, nodes: List[dict]) -> dict:

        """
        Search for a node based on its id.

        Arguments:
            idd (str): node id (double d to not override id built-in function);
            nodes (List[dict]): list of nodes.

        Returns:
            node (dict): found node.
        """

        for node in nodes:
            if node["id"] == idd:
                return node

        raise NodeNotFoundError(f"Node not found: {idd}")

    @staticmethod
    def rebuild(flat_node: dict, nodes: List[dict]) -> Node:

        """
        Rebuild a node (tree structure).

        Arguments:
            node (dict): node as dict to rebuild;
            nodes (List[dict]): list of nodes.

        Returns:
            Node: _description_
        """

        if "label" in flat_node.keys():

            node = LeafNode(
                label=int(flat_node["label"]),
                class_proportions=np.array(flat_node["class_proportions"]),
            )

            return node

        else:

            node = InternalNode(
                feature=flat_node["feature"], threshold=flat_node["threshold"]
            )

            # set children
            if flat_node.get("left", False):
                node.set_left(
                    DecisionTree.rebuild(
                        DecisionTree.search(flat_node["left"], nodes), nodes
                    )
                )

            if flat_node.get("right", False):
                node.set_right(
                    DecisionTree.rebuild(
                        DecisionTree.search(flat_node["right"], nodes), nodes
                    )
                )

            return node

    def load(self, path: str):

        """
        Load model from directory path.

        Arguments:
            path (str): directory's path where the model's parameters are.

        Returns:
            itself.
        """

        with open(f"{path}/tree.json", "r") as f:
            nodes = json.load(f)

        self.tree = DecisionTree.rebuild(nodes[0], nodes)

        return self
