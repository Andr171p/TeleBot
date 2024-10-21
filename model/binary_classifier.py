from pandas import DataFrame
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from model.classifier import Classifier
from model.save import SaveModel


class BinaryClassifierModel(Classifier):
    @classmethod
    def binary_classifier(cls) -> None:
        cls.classifier = RandomForestClassifier()

    @classmethod
    def train(cls, x_train: DataFrame, y_train: DataFrame) -> None:
        cls.classifier.fit(x_train, y_train)

    @classmethod
    def save(cls) -> None:
        save_model = SaveModel(model=cls.classifier)
        save_model.save()

    @classmethod
    def predict(cls, x_test: DataFrame) -> Any:
        y_predict = cls.classifier.predict(x_test)
        return y_predict

    @classmethod
    def model(cls) -> RandomForestClassifier:
        return cls.classifier


# model = RandomForrest()
