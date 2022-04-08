import numpy as np

from src.models.BaseClassifier import BaseClassifier


class Node:
    def __init__(
        self,
        feature: int = None,
        threshold: float = None,
        label: int = None,
        isleaf: bool = None,
    ):

        self.left = None
        self.right = None
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.isleaf = isleaf

    def set_left(self, node):
        self.left = node

    def set_right(self, node):
        self.right = node

    def predict(self, x):

        if not self.label is None:
            return self.label

        if x[self.feature] < self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def inorder(self):

        print(f" feature {self.feature}")
        print(f"threshold {self.threshold}")
        print(f"Label {self.label}")
        if self.left:
            self.left.inorder()
        if self.right:
            self.left.inorder()


class DecisionTree(BaseClassifier):
    def __init__(
        self,
        max_depth: int = None,
        random_state: int = 0,
    ):

        self.max_depth = max_depth
        self.random_state = random_state

    @staticmethod
    def entropy(y):

        _, counts = np.unique(y, return_counts=True)
        p = counts / len(counts)

        return -(p @ np.log(p).T)

    def build_node(
        self, X: np.ndarray, y: np.ndarray, features: np.ndarray, heigth: int = None
    ):

        # no data
        if len(X) == 0:
            return None

        # only one class
        if len(np.unique(y)) == 1:
            return Node(label=np.unique(y)[0], isleaf=True)

        # two or more classes
        else:

            # no more features available - return leaf node using most frequent label
            if len(features) == 0:
                return Node(label=np.argmax(np.bincount(y)), isleaf=True)

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
            node = Node(feature=feature, threshold=splitpoint, isleaf=False)

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

        self.tree = self.build_node(X, y, list(range(X.shape[1])))

        return self

    def predict(self, X):

        return [self.tree.predict(x) for x in X]
