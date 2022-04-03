import logging

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

        logger.info(f"Read provided data: {self.in_file}")
        data = pd.read_csv(f"{self.in_file}")

        # sepal_length,sepal_width,petal_length,petal_width,species

        X = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]].copy()
        y = data["species"].copy()

        logger.info(f"Scale features.")
        # scale features
        scaler = StandardScaler()
        X_preprocessed = scaler.fit_transform(X)

        logger.info(f"Encode string labels to integers.")
        # transform string labels to integers
        encoder = LabelEncoder()
        y_preprocessed = encoder.fit_transform(y)

        # # one hot encode labels
        # enc = OneHotEncoder()
        # y_preprocessed = enc.fit_transform(y.values)

        # store as a dataframe
        df = pd.DataFrame(
            X_preprocessed,
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )

        # append labels column
        df["label"] = y_preprocessed

        df.to_csv(f"data/preprocessed.csv", index=False)

        logger.info(f"Preprocessed data written.")
