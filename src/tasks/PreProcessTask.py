import logging
import pickle

import luigi
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class PreProcessTask(luigi.Task):

    in_file = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f"data/preprocessed.csv")

    def run(self):

        logger.info(f"Start PreProcess Task.")

        # read local data, but we could insert a process
        # to retrieve the data from a private source
        # download it and run the following commands

        logger.info(f"Read provided data: {self.in_file}")
        data = pd.read_csv(f"{self.in_file}")

        X = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]].copy()
        y = data["species"].copy()

        logger.info(f"Scale features.")
        scaler = StandardScaler()
        X_preprocessed = scaler.fit_transform(X)

        logger.info(f"Encode string labels to integers.")
        encoder = LabelEncoder()
        y_preprocessed = encoder.fit_transform(y)

        # store as a dataframe
        df = pd.DataFrame(
            X_preprocessed,
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )

        # append labels column
        df["label"] = y_preprocessed

        df.to_csv(f"data/preprocessed.csv", index=False)

        logger.info(f"Preprocessed data written.")

        # save standard scaler
        pickle.dump(scaler, open("standard-scaler.pkl", "wb"))
