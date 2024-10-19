import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)

from pandas import DataFrame

from typing import Any

from consts import MODEL_INFO_PATH


class Plots:
    def __init__(self, model: Any) -> None:
        self.model = model

    def roc_curve_plot(self, x_test: DataFrame, y_test: DataFrame) -> None:
        y_predict_proba = self.model.predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='orange', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.savefig(fr"{MODEL_INFO_PATH}\roc_curve_plot.png")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
