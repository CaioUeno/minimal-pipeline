import logging

import luigi

from src.tasks.EvaluationTask import EvaluationTask

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

logging.basicConfig(
    filename="automated-eval.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level="INFO",
)


class DeployTask(luigi.Task):

    in_file = luigi.Parameter()

    def complete(self):
        return False

    def requires(self):
        return EvaluationTask(in_file=self.in_file)

    def run(self):

        logger.info(f"Start Deploy Task.")

        # fake deploy
        # nothing to do here since we already have
        # our zip file containing our model
        # and the scaler
