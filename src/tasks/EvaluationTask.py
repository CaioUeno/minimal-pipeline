import logging

import luigi
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from src.models.NaiveBayes import GaussianNaiveBayes
from src.models.NearestNeighbors import NearestNeighbors
from src.tasks.PreProcessTask import PreProcessTask

logger = logging.getLogger(__name__)


class EvaluationTask(luigi.Task):

    in_file = luigi.Parameter()
    best_model_path = luigi.Parameter()

    def requires(self):
        return PreProcessTask(in_file=self.in_file)

    def fit_and_save(self, best_model_name: str, X: np.ndarray, y: np.ndarray):

        if best_model_name == "GaussianNaiveBayes":
            best_model = GaussianNaiveBayes()

        elif best_model_name == "NearestNeighbors":
            best_model = NearestNeighbors(k=5, metric="euclidean", n_jobs=-1)

        else:
            raise ValueError(f"Uknown model: {best_model_name}.")

        best_model.fit(X=X.values, y=y.values)
        # best_model.save(self.model_path)

        return best_model

    def run(self):

        logger.info(f"Start Evaluation Task.")

        logger.info(f"Read preprocessed data: {self.input().path}")
        evaluation_df = pd.read_csv(self.input().path)

        X, y = evaluation_df.drop("label", axis=1), evaluation_df["label"].copy()

        models = [GaussianNaiveBayes(), NearestNeighbors(k=5, metric="euclidean", n_jobs=-1)]

        logger.info(f"")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        metrics = []

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

            # split into train/test
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            for model in models:

                model.fit(X=X_train.values, y=y_train.values)
                preds = model.predict(X=X_test.values)

                metrics.append({
                        "classifier": type(model).__name__,
                        "fold": fold,
                        "f1_score": f1_score(
                            y_true=y_test,
                            y_pred=preds,
                            average="weighted",
                            zero_division=0,
                        ),
                    })


        metrics = pd.DataFrame(metrics)
        # print(metrics)
        avg_performance = metrics.groupby("classifier")["f1_score"].mean()
        best_model_name = avg_performance.idxmax()

        self.fit_and_save(best_model_name, X, y)

        # # one hot encode labels
        # enc = OneHotEncoder()
        # y_preprocessed = enc.fit_transform(y.values)
