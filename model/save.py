import joblib

import pandas as pd

from consts import MODEL_INFO_PATH, TRAINED_MODEL_PATH

from typing import Any

from loguru import logger


class SaveModel:
    COMPRESS: int = 9
    FILENAME: str = TRAINED_MODEL_PATH

    def __init__(self, model: Any) -> None:
        self.model = model

    def save(self) -> None:
        joblib.dump(
            value=self.model,
            filename=self.FILENAME,
            compress=self.COMPRESS
        )
        logger.info(f"Model saves successfully to {self.FILENAME}...")


class SaveMetrics:
    def __init__(
            self, accuracy: float, confusion_matrix: Any, classification_report: Any
    ) -> None:
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report

    def save_accuracy(self) -> None:
        dataframe = pd.DataFrame(
            data=[{
                'accuracy': self.accuracy
            }]
        )
        dataframe.to_csv(fr"{MODEL_INFO_PATH}\accuracy.csv")
        logger.info("Accuracy saved successfully...")

    def save_confusion_matrix(self) -> None:
        dataframe = pd.DataFrame(
            data=self.confusion_matrix,
            columns=['Predicted 0', 'Predicted 1'],
            index=['True 0', 'True 1']
        )
        dataframe.to_csv(fr"{MODEL_INFO_PATH}\confusion_matrix.csv")
        logger.info("Confusion matrix saved successfully...")

    def save_classification_report(self) -> None:
        dataframe = pd.DataFrame(
            data=[{
                'classification_report': self.classification_report
            }]
        )
        dataframe.to_csv(fr"{MODEL_INFO_PATH}\classification_report.csv")
        logger.info("Classification report saves successfully...")

