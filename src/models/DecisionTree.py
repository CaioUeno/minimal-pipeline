import numpy as np

from src.models.BaseClassifier import BaseClassifier


class Node:
    def __init__(self, feature: int = None, threshold: float = None):

        self.left = None
        self.right = None
        self.feature = feature
        self.threshold = threshold
        self.label = None

    def predict(self, x):

        if not self.label is None:
            return self.label

        if x[self.feature < self.threshold]:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


# class Tree:
#     def __init__(self):
#         pass

#     def


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

        return -(p * np.log(p)).sum()

    def build_node(self, X, y):

        if len(X) == 0:
            return None

        if len(np.unique(y)) == 1:
            print(np.unique(y)[0])
            node = Node()
            node.label = np.unique(y)[0]
            return node

        else:

            if len(self.available_features) == 0:
                return None

            global_igs = []
            for feature in self.available_features:

                min_value, max_value, pace = (
                    min(X[:, feature]),
                    max(X[:, feature]),
                    min(np.diff(np.sort(X[:, feature]))),
                )

                f_igs = []
                # information_gain
                for value in np.linspace(min_value, max_value, len(X) + 1):
                    f_igs.append(
                        DecisionTree.entropy(y)
                        - sum(
                            [
                                DecisionTree.entropy(y[X[:, feature] < value]),
                                DecisionTree.entropy(y[X[:, feature] >= value]),
                            ]
                        )
                    )

                global_igs.append(
                    {
                        "feature": feature,
                        "value": np.linspace(min_value, max_value, len(X) + 1)[
                            np.argmax(f_igs)
                        ],
                        "ig": np.max(f_igs),
                    }
                )

            d = max(global_igs, key=lambda item: item["ig"])
            f = d["feature"]
            t = d["value"]
            node = Node(feature=f, threshold=t)
            self.available_features.remove(f)
            node.left = self.build_node(X[X[:, f] < t], y[X[:, f] < t])
            node.right = self.build_node(X[X[:, f] >= t], y[X[:, f] >= t])

        return node

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.available_features = list(range(X.shape[1]))

        self.tree = self.build_node(X, y)

        return self

    def predict(self, X):

        return [self.tree.predict(x) for x in X]
