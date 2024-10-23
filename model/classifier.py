from pandas import DataFrame

from sklearn.linear_model import LogisticRegression

from model.save import SaveModel

from typing import Any


class BinaryClassifierModel:
    classifier: LogisticRegression = None

    @classmethod
    def binary_classifier(cls) -> None:
        cls.classifier = LogisticRegression()

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
    def model(cls) -> LogisticRegression:
        return cls.classifier
