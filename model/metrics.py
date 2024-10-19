from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from typing import Any

from loguru import logger


class Metrics:
    def __init__(self, y_predict: Any, y_test: Any) -> None:
        self.y_predict = y_predict
        self.y_test = y_test

    def accuracy(self) -> float:
        accuracy = accuracy_score(
            y_pred=self.y_predict,
            y_true=self.y_test
        )
        logger.info(f"Model accuracy: {accuracy * 100}%")
        return accuracy

    def confusion_matrix(self) -> list:
        matrix = confusion_matrix(
            y_pred=self.y_predict,
            y_true=self.y_test
        )
        logger.info(f"Model confusion matrix: {matrix}")
        return matrix

    def classification_report(self) -> str | dict:
        report = classification_report(
            y_pred=self.y_predict,
            y_true=self.y_test
        )
        logger.info(f"Model classification report: {report}")
        return report
