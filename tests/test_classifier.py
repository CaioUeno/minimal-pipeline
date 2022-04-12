import os
import numpy as np
import pandas as pd
import pytest

from src.models.DecisionTree import DecisionTree
from src.models.LogisticRegression import LogisticRegression
from src.models.NaiveBayes import GaussianNaiveBayes
from src.models.NearestNeighbors import NearestNeighbors


@pytest.fixture
def data():
    return pd.read_csv("tests/preprocessed.csv")


@pytest.fixture
def models():
    return [
        NearestNeighbors(k=4, metric="euclidean", n_jobs=-1),
        GaussianNaiveBayes(priors=None),
        DecisionTree(max_depth=2),
        LogisticRegression(),
    ]


def test_fit(data, models):

    X = data.drop("label", axis=1).values
    y = data["label"].values

    for model in models:
        model.fit(X, y)


def test_predict(data, models):

    X = data.drop("label", axis=1).values
    y = data["label"].values

    for model in models:

        print(f"{type(model).__name__}")

        pred = model.fit(X, y).predict(X)

        assert isinstance(pred, np.ndarray)
        assert len(X) == len(pred)


def test_predict_proba(data, models):

    X = data.drop("label", axis=1).values
    y = data["label"].values

    for model in models:

        print(f"{type(model).__name__}")

        pred = model.fit(X, y).predict_proba(X)

        assert isinstance(pred, np.ndarray)
        assert len(pred.shape) == 2
        assert len(X) == len(pred)


def test_save(data, models):

    X = data.drop("label", axis=1).values
    y = data["label"].values

    for model in models:

        print(f"{type(model).__name__}")

        model.fit(X, y)

        path = f"{type(model).__name__}"

        model.save(path)

        assert len(os.listdir(path)) > 0
        assert model.compress(path).split("/")[-1] == f"{path}.zip"

        # clean local project
        os.system(f"rm -r {path}")
        os.remove(f"{path}.zip")


def test_load(data, models):

    X = data.drop("label", axis=1).values
    y = data["label"].values

    for model in models:

        print(f"{type(model).__name__}")

        model.fit(X, y)

        preds = model.predict(X)
        preds_proba = model.predict_proba(X)

        path = f"{type(model).__name__}"

        model.save(path)
        model.compress(path)
        model.uncompress(f"{path}.zip")
        model.load(path)

        assert all(preds == model.predict(X))
        assert (preds_proba == model.predict_proba(X)).all()

        # clean local project
        os.system(f"rm -r {path}")
        os.remove(f"{path}.zip")
