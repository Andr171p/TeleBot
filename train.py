from dataset import Dataset

from model.classifier import BinaryClassifierModel
from model.metrics import Metrics
from model.plots import Plots
from model.save import (
    SaveModel,
    SaveMetrics
)

from typing import Any


x_train, y_train, x_test, y_test = Dataset().dataset()


def train_model() -> BinaryClassifierModel.model:
    model = BinaryClassifierModel()
    model.binary_classifier()
    model.train(
        x_train=x_train,
        y_train=y_train
    )
    classifier = model.model()
    save_model = SaveModel(model=classifier)
    save_model.save()
    return model


def predict_model(model: BinaryClassifierModel.model) -> Any:
    y_predict = model.predict(x_test=x_test)
    return y_predict


def metrics(y_predict: Any) -> None:
    model_metrics = Metrics(
        y_predict=y_predict,
        y_test=y_test
    )
    accuracy = model_metrics.accuracy()
    confusion_matrix = model_metrics.confusion_matrix()
    classification_report = model_metrics.classification_report()
    save_metrics = SaveMetrics(
        accuracy=accuracy,
        confusion_matrix=confusion_matrix,
        classification_report=classification_report
    )
    save_metrics.save_accuracy()
    save_metrics.save_confusion_matrix()
    save_metrics.save_classification_report()


def plots(model: BinaryClassifierModel.model) -> None:
    model_plots = Plots(model=model)
    model_plots.roc_curve_plot(
        x_test=x_test, y_test=y_test
    )
