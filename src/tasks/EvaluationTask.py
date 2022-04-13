import logging

import luigi
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from src.models.DecisionTree import DecisionTree
from src.models.LogisticRegression import LogisticRegression
from src.models.NaiveBayes import GaussianNaiveBayes
from src.models.NearestNeighbors import NearestNeighbors
from src.tasks.PreProcessTask import PreProcessTask
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EvaluationTask(luigi.Task):

    in_file = luigi.Parameter()

    def requires(self):
        return PreProcessTask(in_file=self.in_file)

    def output(self):
        return luigi.LocalTarget(f"best_model.zip")

    def fit_and_save(self, best_model_name: str, X: np.ndarray, y: np.ndarray):

        if best_model_name == "GaussianNaiveBayes":
            best_model = GaussianNaiveBayes()

        elif best_model_name == "NearestNeighbors":
            best_model = NearestNeighbors(k=3, metric="euclidean", n_jobs=-1)

        elif best_model_name == "DecisionTree":
            best_model = DecisionTree(max_depth=3)

        elif best_model_name == "LogisticRegression":
            best_model = LogisticRegression()

        else:
            raise ValueError(f"Uknown model: {best_model_name}.")

        best_model.fit(X=X.values, y=y.values)
        best_model.save(f"best_model")

        return best_model.compress(f"best_model")

    def run(self):

        logger.info(f"Start Evaluation Task.")

        logger.info(f"Read preprocessed data: {self.input().path}")
        evaluation_df = pd.read_csv(self.input().path)

        X, y = evaluation_df.drop("label", axis=1), evaluation_df["label"].copy()

        models = [
            GaussianNaiveBayes(),
            NearestNeighbors(k=3, metric="euclidean", n_jobs=-1),
            DecisionTree(max_depth=3),
            LogisticRegression(),
        ]

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        metrics = []

        logger.info(f"Start cross-validation.")
        for fold, (train_index, test_index) in tqdm(
            enumerate(skf.split(X, y)), total=10
        ):

            # split into train/test
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            for model in models:

                model.fit(X=X_train.values, y=y_train.values)
                preds = model.predict(X=X_test.values)

                metrics.append(
                    {
                        "classifier": type(model).__name__,
                        "fold": fold,
                        "f1_score": f1_score(
                            y_true=y_test,
                            y_pred=preds,
                            average="weighted",
                            zero_division=0,
                        ),
                    }
                )

        metrics = pd.DataFrame(metrics)

        avg_performance = metrics.groupby("classifier")["f1_score"].mean()
        avg_performance.round(4).to_csv("report.csv", index=True)
        logger.info(f"Report saved as report.csv")

        best_model_name = avg_performance.idxmax()

        logger.info(f"Best model: {best_model_name}")

        logger.info(f"Fit {best_model_name} using entire data, save  it and compress.")
        self.fit_and_save(best_model_name, X, y)
